# Sports Video Generation (Wan-style baseline)

This repo contains a compact Wan-style video generation pipeline built with PyTorch:

- Causal 3D VAE for video latent compression/reconstruction
- Diffusion Transformer (DiT) with T5 text conditioning
- DDPM training objective + classifier-free guidance sampling
- Train/infer scripts wired for SoccerTrack data workflows

## Data pipeline (SoccerTrack v2)

We use the pre-labeled SoccerTrack v2 dataset from Hugging Face as the source corpus and prepare model-ready clips locally in `soccertrack_data/`.

Expected prep flow:

1. Download SoccerTrack v2 games and metadata from Hugging Face.
2. Segment each game into short fixed windows (default: 5-second clips).
3. Save each clip as a standalone video file under `soccertrack_data/` (nested folders are fine).
4. Optionally add a same-name `.txt` caption next to each clip for text conditioning.

Example layout:

```text
soccertrack_data/
  game_001/
    clip_000001.mp4
    clip_000001.txt
    clip_000002.mp4
  game_002/
    clip_000101.mp4
```

If a `.txt` caption is missing, the scripts fall back to a cleaned filename as the prompt.

## Files

- `wan_model.py`: model components (VAE, T5 wrapper, DiT blocks, DDPM schedule, CFG sampler)
- `train.py`: training loop with 80/20 train/validation split from `soccertrack_data/`
- `infer.py`: checkpoint-based sampling; can use explicit prompt or sample prompt from `soccertrack_data/`

## Training

```bash
python train.py \
  --data_dir ./soccertrack_data \
  --epochs 10 \
  --batch_size 2
```

Notes:

- Input videos are resized to the configured resolution and normalized to `[-1, 1]`.
- Validation runs every epoch on the 20% split.
- Checkpoints are saved to `checkpoints/`.

## Inference

Use an explicit prompt:

```bash
python infer.py \
  --checkpoint checkpoints/wan_epoch_010.pt \
  --prompt "broadcast wide shot, fast counterattack through midfield" \
  --output sample.mp4
```

Or pull a prompt from `soccertrack_data/` captions/filenames:

```bash
python infer.py \
  --checkpoint checkpoints/wan_epoch_010.pt \
  --prompt_from_data \
  --data_dir ./soccertrack_data \
  --output sample.mp4
```

## Dependencies

Core:

- `torch`
- `torchvision` (video loading)
- `transformers` (T5 encoder)
- `Pillow`
- `imageio`
- `numpy`

CUDA is used automatically when available.
