#!/usr/bin/env python3
import argparse
import os

import torch
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video


def chunked_generate(
    pipe: WanVideoPipeline,
    input_frames,
    prompt: str,
    negative_prompt: str,
    total_frames: int,
    chunk_frames: int,
    overlap: int,
    denoising_strength: float,
    seed: int,
    height: int,
    width: int,
):
    if chunk_frames <= overlap:
        raise ValueError("chunk_frames must be greater than overlap")
    if len(input_frames) < chunk_frames:
        raise ValueError("input clip is shorter than chunk_frames")

    generated = []
    chunk_idx = 0
    stride = chunk_frames - overlap

    while len(generated) < total_frames:
        if chunk_idx == 0:
            cond_frames = input_frames[:chunk_frames]
        else:
            # Use the tail of generated output as conditioning for temporal continuity.
            tail = generated[-overlap:]
            remaining = chunk_frames - overlap
            src_start = min(chunk_idx * stride, max(0, len(input_frames) - remaining))
            src_pad = input_frames[src_start : src_start + remaining]
            if len(src_pad) < remaining:
                src_pad = src_pad + [input_frames[-1]] * (remaining - len(src_pad))
            cond_frames = tail + src_pad

        chunk_video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_video=cond_frames,
            denoising_strength=denoising_strength,
            seed=seed + chunk_idx,
            tiled=True,
            num_frames=chunk_frames,
            height=height,
            width=width,
        )

        if chunk_idx == 0:
            generated.extend(chunk_video)
        else:
            generated.extend(chunk_video[overlap:])

        chunk_idx += 1
        if len(generated) >= total_frames:
            break

    return generated[:total_frames]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_clip", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default="blurry, low quality, artifacts, deformed players, watermark, text")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=336)
    parser.add_argument("--chunk_frames", type=int, default=17)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--denoising_strength", type=float, default=0.3)
    parser.add_argument("--lora_alpha", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    total_frames = args.fps * args.seconds
    if total_frames < args.chunk_frames:
        total_frames = args.chunk_frames

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    source = VideoData(video_file=args.input_clip, height=args.height, width=args.width)
    source.set_length(max(args.chunk_frames, total_frames))
    input_frames = source.raw_data()

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        redirect_common_files=False,
    )
    pipe.load_lora(pipe.dit, args.lora_path, alpha=args.lora_alpha)

    frames = chunked_generate(
        pipe=pipe,
        input_frames=input_frames,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        total_frames=total_frames,
        chunk_frames=args.chunk_frames,
        overlap=args.overlap,
        denoising_strength=args.denoising_strength,
        seed=args.seed,
        height=args.height,
        width=args.width,
    )
    save_video(frames, args.output_path, fps=args.fps, quality=5)
    print(f"saved {args.output_path}")


if __name__ == "__main__":
    main()

