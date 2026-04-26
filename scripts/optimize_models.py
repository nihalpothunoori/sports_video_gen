#!/usr/bin/env python3
"""
Convert Wan2.1 model files to FP16 on disk.

For BF16 sources, this preserves raw parameter size because both formats use
16 bits per value. FP32 sources will shrink by about half.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def cast_to_fp16(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.to(torch.float16)
    if isinstance(value, dict):
        return {key: cast_to_fp16(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [cast_to_fp16(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cast_to_fp16(item) for item in value)
    return value


def cast_state_dict_to_fp16_in_place(state_dict: dict[str, Any]) -> dict[str, Any]:
    for key, value in list(state_dict.items()):
        if torch.is_tensor(value):
            state_dict[key] = value.to(torch.float16)
        elif isinstance(value, dict):
            state_dict[key] = cast_state_dict_to_fp16_in_place(value)
        elif isinstance(value, list):
            state_dict[key] = [cast_to_fp16(item) for item in value]
        elif isinstance(value, tuple):
            state_dict[key] = tuple(cast_to_fp16(item) for item in value)
    return state_dict


def cast_torch_checkpoint_to_fp16(src: Path, dst: Path) -> None:
    print(f"\n[1/2] Converting T5 checkpoint:\n  src={src}\n  dst={dst}")
    load_kwargs = {"map_location": "cpu"}
    if "mmap" in torch.load.__code__.co_varnames:
        load_kwargs["mmap"] = True
    checkpoint = torch.load(src, **load_kwargs)
    if isinstance(checkpoint, dict):
        checkpoint = cast_state_dict_to_fp16_in_place(checkpoint)
    else:
        checkpoint = cast_to_fp16(checkpoint)
    torch.save(checkpoint, dst)
    del checkpoint
    gc.collect()


def cast_safetensors_to_fp16(src: Path, dst: Path) -> None:
    print(f"\n[2/2] Converting DiT safetensors:\n  src={src}\n  dst={dst}")
    tensors = load_file(str(src), device="cpu")
    for key, value in list(tensors.items()):
        tensors[key] = value.to(torch.float16)
    save_file(tensors, str(dst))
    del tensors
    gc.collect()


def report_file_size_change(src: Path, dst: Path) -> None:
    src_size = src.stat().st_size
    dst_size = dst.stat().st_size
    delta = dst_size - src_size
    sign = "+" if delta >= 0 else "-"
    print(
        f"  {src.name} -> {dst.name}: "
        f"{human_size(src_size)} -> {human_size(dst_size)} "
        f"(delta {sign}{human_size(abs(delta))})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BF16 Wan2.1 model weights to FP16 on disk."
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("/home/nihal/Documents/cachehacks/models/wan2.1-1.3b"),
        help="Directory containing Wan2.1 model files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing FP16 output files if they already exist.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    t5_src = model_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    dit_src = model_dir / "diffusion_pytorch_model.safetensors"
    vae_path = model_dir / "Wan2.1_VAE.pth"

    t5_dst = model_dir / "models_t5_umt5-xxl-enc-fp16.pth"
    dit_dst = model_dir / "diffusion_pytorch_model_fp16.safetensors"

    required = [t5_src, dit_src, vae_path]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s):\n" + "\n".join(f"  - {p}" for p in missing)
        )

    outputs = [t5_dst, dit_dst]
    existing_outputs = [p for p in outputs if p.exists()]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output file(s). Pass --overwrite to replace:\n"
            + "\n".join(f"  - {p}" for p in existing_outputs)
        )

    cast_torch_checkpoint_to_fp16(t5_src, t5_dst)
    cast_safetensors_to_fp16(dit_src, dit_dst)

    print("\nSize report:")
    report_file_size_change(t5_src, t5_dst)
    report_file_size_change(dit_src, dit_dst)

    model_paths = [
        str(dit_dst),
        str(t5_dst),
        str(vae_path),
    ]
    print("\nUpdated --model_paths JSON:")
    print(json.dumps(model_paths))
    print(
        "\nNote: BF16 -> FP16 keeps the same raw parameter size, while FP32 -> FP16 "
        "cuts it roughly in half."
    )


if __name__ == "__main__":
    main()
