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
# Enable TF32 for A40 (Ampere)
torch.set_float32_matmul_precision('high')
# Prevent Dynamo recompile limit crashes
torch._dynamo.config.recompile_limit = 128

def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def generate(args):
    # Prompt and Parameters
    prompt = "realistic filming style, a person wearing a dark helmet, a deep-colored jacket, blue jeans, and bright yellow shoes rides a skateboard along a winding mountain road. The skateboarder starts in a standing position, then gradually lowers into a crouch, extending one hand to touch the road surface while maintaining a low center of gravity to navigate a sharp curve. After completing the turn, the skateboarder rises back to a standing position and continues gliding forward. The background features lush green hills flanking both sides of the road, with distant snow-capped mountain peaks rising against a clear, bright blue sky. The camera follows closely from behind, smoothly tracking the skateboarder’s movements and capturing the dynamic scenery along the route. The scene is shot in natural daylight, highlighting the vivid outdoor environment and the skateboarder’s fluid actions."
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    num_segments = 11
    num_frames = 61
    num_cond_frames = 13
    spatial_refine_only = False

    checkpoint_dir = args.checkpoint_dir
    context_parallel_size = args.context_parallel_size
    enable_compile = args.enable_compile

    # Prepare Distributed Env
    rank = int(os.environ.get('RANK', 0))
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))

    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()

    init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank,
                          world_size=num_processes)
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    # 1. LOAD MODELS
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder",
                                                    torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler",
                                                                torch_dtype=torch.bfloat16)
    dit = LongCatVideoTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw,
                                                         torch_dtype=torch.bfloat16)

    # 2. COMPILE STRATEGY
    if enable_compile:
        # Use dynamic=True because video segments can vary in internal shape/trace
        dit = torch.compile(dit, dynamic=True)

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )

    # Send main models to GPU
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

    # Offload text encoder to save 10GB+ VRAM for the continuation loop
    pipe.text_encoder.to("cpu")
    torch_gc()

    if local_rank == 0:
        output_tensor = torch.from_numpy(np.array(output))
        output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
        write_video(f"output_long_video_initial.mp4", output_tensor, fps=15, video_codec="libx264")

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

        # FIX: Move text_encoder back to GPU before generation
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

        # FIX: Move back to CPU after each segment to keep VRAM free for KV cache
        pipe.text_encoder.to("cpu")
        torch_gc()

        new_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        new_video = [PIL.Image.fromarray(img) for img in new_video]
        all_generated_frames.extend(new_video[num_cond_frames:])
        current_video = new_video
        del output
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

        # Select the chunk of the original video to refine
        chunk = all_generated_frames[start_id:start_id + num_frames]

        # Refinement doesn't always need the text encoder, but let's ensure it's where it needs to be
        # based on the LongCat pipeline's requirements for 'generate_refine'
        pipe.text_encoder.to(local_rank)

        output_refine = pipe.generate_refine(
            video=cur_condition_video,
            prompt='',
            stage1_video=chunk,
            num_cond_frames=cur_num_cond_frames,
            num_inference_steps=50,
            generator=generator,
            spatial_refine_only=spatial_refine_only
        )[0]

        pipe.text_encoder.to("cpu") # Keep it on CPU while not in use

        new_video = [(output_refine[i] * 255).astype(np.uint8) for i in range(output_refine.shape[0])]
        new_video = [PIL.Image.fromarray(img) for img in new_video]
        del output_refine

        all_refine_frames.extend(new_video[cur_num_cond_frames:])
        cur_condition_video = new_video
        cur_num_cond_frames = num_frames if spatial_refine_only else num_frames * 2
        start_id = start_id + num_frames - num_cond_frames

        if local_rank == 0 and segment_idx % 2 == 0:
            output_tensor = torch.from_numpy(np.array(all_refine_frames))
            fps = 15 if spatial_refine_only else 30
            write_video(f"output_refine_partial_{segment_idx}.mp4", output_tensor, fps=fps, video_codec="libx264")

        torch_gc()

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_parallel_size", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument('--enable_compile', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    generate(args)