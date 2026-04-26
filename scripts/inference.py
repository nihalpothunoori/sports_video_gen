#!/usr/bin/env python3
"""
Wan2.1 T2V 1.3B inference — base model or LoRA.

Usage:
  python scripts/inference.py "pass"
  python scripts/inference.py "shot" --lora_path ./lora_output/

Saves to ./output/counterfactual_{timestamp}.mp4
"""

import sys, os, argparse, glob
from datetime import datetime
from pathlib import Path

# DiffSynth-Studio must be on the path
sys.path.insert(0, str(Path(__file__).parent / 'DiffSynth-Studio'))

import torch
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video

MODEL_DIR   = Path('/home/nihal/Documents/cachehacks/models/wan2.1-1.3b')
OUTPUT_DIR  = Path('/home/nihal/Documents/cachehacks/output')

# Standard quality negative prompt used in all official Wan2.1 examples
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def find_lora(lora_path: Path) -> str | None:
    """Return the most recently modified .safetensors in lora_path, or None."""
    candidates = sorted(
        lora_path.glob('*.safetensors'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def build_pipeline() -> WanVideoPipeline:
    """
    DiT=5.3GB + T5=11GB + VAE=0.5GB = 16.8GB total — exceeds 16GB VRAM.
    Use CPU offload: idle models sit in CPU RAM (21GB available), move to
    GPU only when computing. vram_limit tells DiffSynth how much VRAM to
    use before offloading; 13GB leaves room for activations.
    """
    vram_config = {
        "offload_dtype":      torch.bfloat16,
        "offload_device":     "cpu",
        "onload_dtype":       torch.bfloat16,
        "onload_device":      "cpu",
        "preparing_dtype":    torch.bfloat16,
        "preparing_device":   "cuda",
        "computation_dtype":  torch.bfloat16,
        "computation_device": "cuda",
    }

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=str(MODEL_DIR / 'diffusion_pytorch_model.safetensors'), **vram_config),
            ModelConfig(path=str(MODEL_DIR / 'models_t5_umt5-xxl-enc-bf16.pth'),    **vram_config),
            ModelConfig(path=str(MODEL_DIR / 'Wan2.1_VAE.pth'),                     **vram_config),
        ],
        tokenizer_config=ModelConfig(
            path=str(MODEL_DIR / 'google' / 'umt5-xxl'),
        ),
        vram_limit=13,  # GB — keeps each stage within VRAM, offloads others to CPU
    )
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str, help='Text prompt (e.g. "pass", "shot")')
    parser.add_argument('--lora_path', type=str, default='./lora_output',
                        help='Directory containing LoRA .safetensors files')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── LoRA checkpoint ──────────────────────────────────────────────────────
    lora_path = Path(args.lora_path)
    lora_ckpt = find_lora(lora_path)
    if lora_ckpt:
        print(f'LoRA found: {lora_ckpt}')
    else:
        print('No LoRA found, using base model')

    # ── Pipeline ─────────────────────────────────────────────────────────────
    print('Loading pipeline...')
    pipe = build_pipeline()

    if lora_ckpt:
        print(f'Loading LoRA weights...')
        pipe.load_lora(pipe.dit, lora_ckpt, alpha=1)

    # ── Generate ─────────────────────────────────────────────────────────────
    print(f'Generating: "{args.prompt}"')
    video = pipe(
        prompt=args.prompt,
        negative_prompt=NEGATIVE_PROMPT,
        seed=42,
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=6.0,
        num_inference_steps=50,
        tiled=True,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = OUTPUT_DIR / f'counterfactual_{timestamp}.mp4'
    save_video(video, str(out_path), fps=16, quality=5)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
