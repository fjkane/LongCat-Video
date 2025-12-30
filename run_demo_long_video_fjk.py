import os
import argparse
import datetime
import PIL.Image
import numpy as np
import torch
import torch.distributed as dist
import torch._dynamo
from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel

# --- GLOBAL CONFIGS ---
# Set memory management to avoid fragmentation on A40
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision('high')
torch._dynamo.config.recompile_limit = 128


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def generate(args):
    # 1. SETUP PATHS AND DIRECTORIES
    output_dir = args.output_dir
    checkpoint_dir = args.checkpoint_dir

    rank = int(os.environ.get('RANK', 0))
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))

    if local_rank == 0 and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    prompt = "In a realistic filming style, a man in a casual linen shirt sits at a sun-drenched wooden table featuring a ceramic coffee cup, a porcelain creamer pot, a flaky golden croissant, and a glass jar of orange marmalade. The scene begins with the man pouring cream into his coffee, creating swirling patterns, before stirring it gently with a silver spoon. He then picks up the croissant to take a large bite, sets it down, and uses a butter knife to spread a thick layer of glistening marmalade onto the bitten interior surface before taking another bite. The video continues in a steady, rhythmic pace, showing the man repeating this cycle of biting and spreading until the croissant is completely consumed, leaving only golden crumbs on the plate. The camera maintains a focused, close-up perspective with soft natural lighting, capturing the tactile textures of the pastry and the steam rising from the coffee in a calm, continuous sequence."
    negative_prompt = "Bright tones, overexposed, static, blurred details..."

    num_segments = 11
    num_frames = 61
    num_cond_frames = 13
    spatial_refine_only = False
    context_parallel_size = args.context_parallel_size
    enable_compile = args.enable_compile

    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()

    init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank,
                          world_size=num_processes)
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    # 2. LOAD MODELS
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder",
                                                    torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler",
                                                                torch_dtype=torch.bfloat16)
    dit = LongCatVideoTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw,
                                                         torch_dtype=torch.bfloat16)

    if enable_compile:
        dit = torch.compile(dit)

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )

    pipe.to(local_rank)

    # 3. INITIAL T2V GENERATION (480p)
    generator = torch.Generator(device=local_rank)
    generator.manual_seed(42 + global_rank)

    output = pipe.generate_t2v(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_frames=num_frames,
        num_inference_steps=50,
        guidance_scale=4.0,
        generator=generator,
    )[0]

    pipe.text_encoder.to("cpu")
    torch_gc()

    if local_rank == 0:
        output_tensor = torch.from_numpy(np.array(output))
        output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
        initial_path = os.path.join(output_dir, "output_long_video_initial.mp4")
        write_video(initial_path, output_tensor, fps=15, video_codec="libx264")

    video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
    video = [PIL.Image.fromarray(img) for img in video]
    del output
    torch_gc()

    # 4. VIDEO CONTINUATION LOOP
    all_generated_frames = video
    current_video = video

    for segment_idx in range(num_segments):
        if local_rank == 0:
            print(f"Generating segment {segment_idx + 1}/{num_segments}...")

        pipe.text_encoder.to(local_rank)
        output = pipe.generate_vc(
            video=current_video,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution='480p',
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=50,
            guidance_scale=4.0,
            generator=generator,
            use_kv_cache=True,
            offload_kv_cache=True,
            enhance_hf=True
        )[0]

        pipe.text_encoder.to("cpu")
        torch_gc()

        new_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        new_video = [PIL.Image.fromarray(img) for img in new_video]
        all_generated_frames.extend(new_video[num_cond_frames:])
        current_video = new_video
        del output
        torch_gc()

    # --- FIX 1: CLEAR EVERYTHING BEFORE REFINEMENT ---
    if local_rank == 0:
        print("Clearing cache for Refinement Stage...")

    # Reset KV cache explicitly if the pipeline has a reset method
    if hasattr(pipe.dit, "reset_kv_cache"):
        pipe.dit.reset_kv_cache()

    torch_gc()

    # 5. REFINEMENT STAGE (720p)
    if local_rank == 0:
        print("Starting Refinement Stage...")

    refinement_lora_path = os.path.join(checkpoint_dir, 'lora/refinement_lora.safetensors')
    pipe.dit.load_lora(refinement_lora_path, 'refinement_lora')
    pipe.dit.enable_loras(['refinement_lora'])
    pipe.dit.enable_bsa()

    cur_condition_video = None
    cur_num_cond_frames = 0
    start_id = 0
    all_refine_frames = []

    for segment_idx in range(num_segments + 1):
        if local_rank == 0:
            print(f"Refining segment {segment_idx + 1}/{num_segments + 1}...")

        chunk = all_generated_frames[start_id:start_id + num_frames]

        # FIX 2: During VAE encoding (the crash point), offload DIT to CPU
        # to ensure the VAE has maximum VRAM for the 720p sequence
        pipe.dit.to("cpu")
        pipe.text_encoder.to(local_rank)
        torch_gc()

        output_refine = pipe.generate_refine(
            video=cur_condition_video,
            prompt='',
            stage1_video=chunk,
            num_cond_frames=cur_num_cond_frames,
            num_inference_steps=50,
            generator=generator,
            spatial_refine_only=spatial_refine_only
        )[0]

        # Bring DIT back for next iteration if needed (or let the pipeline handle)
        pipe.dit.to(local_rank)
        pipe.text_encoder.to("cpu")

        new_video = [(output_refine[i] * 255).astype(np.uint8) for i in range(output_refine.shape[0])]
        new_video = [PIL.Image.fromarray(img) for img in new_video]
        del output_refine

        all_refine_frames.extend(new_video[cur_num_cond_frames:])
        cur_condition_video = new_video
        cur_num_cond_frames = num_frames if spatial_refine_only else num_frames * 2
        start_id = start_id + num_frames - num_cond_frames

        if local_rank == 0 and segment_idx % 2 == 0:
            output_tensor = torch.from_numpy(np.array(all_refine_frames))
            fps = 30
            refine_path = os.path.join(output_dir, f"output_refine_partial_{segment_idx}.mp4")
            write_video(refine_path, output_tensor, fps=fps, video_codec="libx264")

        torch_gc()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_parallel_size", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument('--enable_compile', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(args)