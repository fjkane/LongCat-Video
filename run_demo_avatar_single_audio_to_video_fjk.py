import os
import json
import time
import math
import random
import argparse
import datetime
import PIL.Image
import numpy as np
from pathlib import Path

import torch
import torch.distributed as dist

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers.utils import load_image

from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video.context_parallel import context_parallel_util

# -------- avatar related --------
import librosa
from longcat_video.audio_process.wav2vec2 import Wav2Vec2ModelWrapper
from longcat_video.audio_process.torch_utils import save_video_ffmpeg
from transformers import Wav2Vec2FeatureExtractor
from audio_separator.separator import Separator


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def generate_random_uid():
    timestamp_part = str(int(time.time()))[-6:]
    random_part = str(random.randint(100000, 999999))
    uid = timestamp_part + random_part
    return uid


def extract_vocal_from_speech(source_path, target_path, vocal_separator, audio_output_dir_temp):
    outputs = vocal_separator.separate(source_path)
    if len(outputs) <= 0:
        return None
    default_vocal_path = audio_output_dir_temp / "vocals" / outputs[0]
    default_vocal_path = default_vocal_path.resolve().as_posix()
    os.system(f"mv '{default_vocal_path}' '{target_path}'")
    return target_path


def generate(args):
    # Load args
    input_json, checkpoint_dir = args.input_json, args.checkpoint_dir
    context_parallel_size = args.context_parallel_size
    stage_1, num_inference_steps = args.stage_1, args.num_inference_steps
    text_guidance_scale, audio_guidance_scale = args.text_guidance_scale, args.audio_guidance_scale
    resolution, num_segments, output_dir = args.resolution, max(1, args.num_segments), args.output_dir

    save_fps, num_frames, num_cond_frames, audio_stride = 16, 51, 13, 2
    height, width = (480, 832) if resolution == '480p' else (768, 1280)

    with open(input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    prompt = input_data['prompt']
    negative_prompt = "Close-up, Bright tones, overexposed, static, blurred details..."  # Truncated for brevity
    raw_speech_path = input_data['cond_audio']['person1']

    # Prepare Dist
    rank = int(os.environ.get('RANK', 0))
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))

    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()

    context_parallel_util.init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank,
                                                world_size=num_processes)
    cp_rank, cp_size = context_parallel_util.get_cp_rank(), context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    # 1. INITIALIZE AUDIO MODELS (Run first, then offload)
    wav2vec_path = os.path.join(checkpoint_dir, 'chinese-wav2vec2-base')
    audio_encoder = Wav2Vec2ModelWrapper(wav2vec_path).to(local_rank)
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path, local_files_only=True)

    # 2. INITIALIZE CORE MODELS
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'),
                                              subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'),
                                                    subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'), subfolder="vae",
                                           torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(checkpoint_dir, '..', 'LongCat-Video'),
                                                                subfolder="scheduler", torch_dtype=torch.bfloat16)
    dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="avatar_single",
                                                               cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)

    pipe = LongCatVideoAvatarPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
        scheduler=scheduler, dit=dit, audio_encoder=audio_encoder,
        wav2vec_feature_extractor=wav2vec_feature_extractor
    )
    pipe.to(local_rank)

    # 3. AUDIO PROCESSING & OFFLOADING
    if cp_rank == 0:
        # (Vocal separation logic...)
        vocal_separator_path = os.path.join(checkpoint_dir, 'vocal_separator/Kim_Vocal_2.onnx')
        audio_output_dir_temp = Path(f"./audio_temp_file")
        os.makedirs(audio_output_dir_temp, exist_ok=True)
        vocal_separator = Separator(output_dir=audio_output_dir_temp / "vocals",
                                    model_file_dir=os.path.dirname(vocal_separator_path))
        vocal_separator.load_model(os.path.basename(vocal_separator_path))

        temp_vocal_path = extract_vocal_from_speech(raw_speech_path, f"/tmp/vocal_{generate_random_uid()}.wav",
                                                    vocal_separator, audio_output_dir_temp)
        speech_array, sr = librosa.load(temp_vocal_path, sr=16000)

        # Audio Embedding
        full_audio_emb = pipe.get_audio_embedding(speech_array, fps=save_fps * audio_stride, device=local_rank,
                                                  sample_rate=sr)

        # CRITICAL: Offload Audio Encoder immediately to free ~2GB VRAM
        pipe.audio_encoder.to("cpu")
        torch_gc()

        if context_parallel_util.get_cp_size() > 1:
            context_parallel_util.cp_broadcast(torch.tensor(list(full_audio_emb.size()), device=local_rank))
            context_parallel_util.cp_broadcast(full_audio_emb)
    else:
        # Receive broadcast... (omitted for brevity)
        pass

    # 4. TEXT ENCODING & OFFLOADING
    # The first generation call usually triggers text encoding.
    # To save A40 memory, we generate the first segment then move T5 to CPU.

    # (Prep first audio_emb indices...)
    indices = torch.arange(5) - 2
    audio_emb = full_audio_emb[
        torch.clamp(torch.arange(0, audio_stride * num_frames, audio_stride).unsqueeze(1) + indices, 0,
                    full_audio_emb.shape[0] - 1)][None, ...].to(local_rank)

    # Generate First Segment
    generator = torch.Generator(device=local_rank).manual_seed(42 + global_rank)

    if stage_1 == 'at2v':
        output_tuple = pipe.generate_at2v(prompt=prompt, height=height, width=width, num_frames=num_frames,
                                          audio_emb=audio_emb, generator=generator)
    else:
        image = load_image(input_data['cond_image'])
        output_tuple = pipe.generate_ai2v(image=image, prompt=prompt, num_frames=num_frames, audio_emb=audio_emb,
                                          generator=generator)

    # CRITICAL: Offload Text Encoder to free ~12GB VRAM
    pipe.text_encoder.to("cpu")
    torch_gc()

    output, latent = output_tuple
    video = [PIL.Image.fromarray((output[0][i] * 255).astype(np.uint8)) for i in range(output[0].shape[0])]

    # 5. LONG VIDEO CONTINUATION
    all_generated_frames = video
    current_video = video
    ref_latent = latent[:, :, :1].clone()

    for segment_idx in range(1, num_segments):
        # Update audio_emb for next segment...
        audio_start_idx = audio_stride * (num_frames - num_cond_frames) * segment_idx
        center_indices = torch.clamp(
            torch.arange(audio_start_idx, audio_start_idx + audio_stride * num_frames, audio_stride).unsqueeze(
                1) + indices, 0, full_audio_emb.shape[0] - 1)
        audio_emb = full_audio_emb[center_indices][None, ...].to(local_rank)

        output_tuple = pipe.generate_avc(
            video=current_video, video_latent=latent, prompt=prompt,
            num_frames=num_frames, num_cond_frames=num_cond_frames,
            use_kv_cache=True,
            offload_kv_cache=True,  # Critical for A40
            audio_emb=audio_emb, ref_latent=ref_latent
        )

        output, latent = output_tuple
        new_video = [PIL.Image.fromarray((output[0][i] * 255).astype(np.uint8)) for i in range(output[0].shape[0])]
        all_generated_frames.extend(new_video[num_cond_frames:])
        current_video = new_video

        if cp_rank == 0:
            save_video_ffmpeg(torch.from_numpy(np.array(all_generated_frames)),
                              os.path.join(output_dir, f"segment_{segment_idx}"), raw_speech_path, fps=save_fps)
        torch_gc()


def _parse_args():
    # ... (same as your original parse_args)
    pass


if __name__ == "__main__":
    generate(_parse_args())