import os
import argparse
import datetime
import PIL.Image
import numpy as np
import torch
import torch.distributed as dist
import torch._dynamo
from transformers import AutoTokenizer, UMT5EncoderModel
from transformers.utils import move_cache
from torchvision.io import write_video

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel

# --- GLOBAL STABILITY CONFIGS ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision('high')
torch._dynamo.config.recompile_limit = 128


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def save_segment(frames, folder, filename, fps=15):
    """
    Robustly saves frames to video.
    Expects frames as a list of numpy arrays [H, W, C] in range [0, 1] or [0, 255].
    """
    output_path = os.path.join(folder, filename)

    # Convert list of frames to a single numpy array [T, H, W, C]
    video_np = np.stack([np.array(f) for f in frames], axis=0)

    # Ensure range is 0-255 uint8
    if video_np.max() <= 1.1:  # Likely float [0, 1]
        video_np = (video_np * 255).astype(np.uint8)
    else:
        video_np = video_np.astype(np.uint8)

    # Final check for Black Frames: if mean is 0, the model failed to generate
    if video_np.mean() < 0.1:
        print(f"WARNING: Segment {filename} appears to be black. Check Model/Dtype stability.")

    tensor_frames = torch.from_numpy(video_np)
    write_video(output_path, tensor_frames, fps=fps, video_codec="libx264")
    return filename


def generate(args):
    try:
        move_cache()
    except Exception:
        pass

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

    # 1. LOAD MODELS
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder",
                                                    torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler",
                                                                torch_dtype=torch.bfloat16)
    dit = LongCatVideoTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="dit", torch_dtype=torch.bfloat16)

    # Enable VAE tiling to prevent OOM
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()

    if args.enable_compile:
        dit = torch.compile(dit)

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to(local_rank)

    # Parameters
    num_segments = 11
    num_frames = 61
    num_cond_frames = 13
    spatial_refine_only = False
    generator = torch.Generator(device=local_rank).manual_seed(42 + dist.get_rank())

    # 2. INITIAL T2V GENERATION
    prompt = "realistic filming style, a person wearing a dark helmet, a deep-colored jacket, blue jeans, and bright yellow shoes rides a skateboard..."
    negative_prompt = "Bright tones, overexposed, static, blurred details..."

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

    # Critical: Offload text encoder to save 10GB RAM
    pipe.text_encoder.to("cpu")
    torch_gc()

    if local_rank == 0:
        save_segment(output, output_dir, "segment_00_initial.mp4")

    # Convert to PIL/List for consistency
    all_generated_frames = []
    for f in output:
        img_np = (f * 255).astype(np.uint8) if f.max() <= 1.1 else f.astype(np.uint8)
        all_generated_frames.append(PIL.Image.fromarray(img_np))

    current_video = all_generated_frames
    del output
    torch_gc()

    # 3. VIDEO CONTINUATION LOOP
    for segment_idx in range(num_segments):
        if local_rank == 0:
            print(f"Continuation Segment {segment_idx + 1}...")

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

        new_video_pil = []
        for f in output:
            img_np = (f * 255).astype(np.uint8) if f.max() <= 1.1 else f.astype(np.uint8)
            new_video_pil.append(PIL.Image.fromarray(img_np))

        if local_rank == 0:
            save_segment(new_video_pil, output_dir, f"segment_{segment_idx + 1:02d}_continuation.mp4")

        all_generated_frames.extend(new_video_pil[num_cond_frames:])
        current_video = new_video_pil
        del output
        torch_gc()

    # 4. REFINEMENT STAGE
    if hasattr(pipe.dit, "reset_kv_cache"):
        pipe.dit.reset_kv_cache()
    torch_gc()

    refinement_lora_path = os.path.join(checkpoint_dir, 'lora/refinement_lora.safetensors')
    pipe.dit.load_lora(refinement_lora_path, 'refinement_lora')
    pipe.dit.enable_loras(['refinement_lora'])
    pipe.dit.enable_bsa()

    cur_condition_video = None
    cur_num_cond_frames = 0
    start_id = 0
    refined_filenames = []

    for segment_idx in range(num_segments + 1):
        if local_rank == 0:
            print(f"Refinement Segment {segment_idx} (720p)...")

        chunk = all_generated_frames[start_id:start_id + num_frames]

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

        pipe.dit.to(local_rank)
        pipe.text_encoder.to("cpu")

        new_video_pil = []
        for f in output_refine:
            img_np = (f * 255).astype(np.uint8) if f.max() <= 1.1 else f.astype(np.uint8)
            new_video_pil.append(PIL.Image.fromarray(img_np))

        if local_rank == 0:
            fname = f"refine_{segment_idx:02d}.mp4"
            save_segment(new_video_pil, output_dir, fname, fps=15)
            refined_filenames.append(fname)

        cur_condition_video = new_video_pil
        cur_num_cond_frames = num_frames if spatial_refine_only else num_frames * 2
        start_id = start_id + num_frames - num_cond_frames

        del output_refine
        torch_gc()

    if local_rank == 0:
        txt_path = os.path.join(output_dir, "inputs.txt")
        with open(txt_path, "w") as f:
            for f_name in refined_filenames:
                f.write(f"file '{f_name}'\n")
        print(f"\n[DONE] ffmpeg -f concat -safe 0 -i {txt_path} -c copy final_video.mp4")


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