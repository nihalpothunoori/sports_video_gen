# Sports Video Generation (Wan-style baseline)

This repo contains a compact Wan-style video generation pipeline built with PyTorch:

- Causal 3D VAE for video latent compression/reconstruction
- Diffusion Transformer (DiT) with T5 text conditioning
- DDPM training objective + classifier-free guidance sampling
- NLP prompt mapper that normalizes free text into SoccerTrack v2 action vocabulary
- Train/infer scripts wired for SoccerTrack data workflows

## Demo video

- [Project demo on YouTube](https://www.youtube.com/watch?v=ES00rbzp4JY)

## Data pipeline (SoccerTrack v2)

We use the pre-labeled SoccerTrack v2 dataset from Hugging Face as the source corpus and prepare model-ready clips locally in `soccertrack_data/`.
It's like a hundred gigabytes of data so I haven't included it, but the format of the clips are below.

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

### SoccerTrack v2 action vocabulary used by the mapper

The NLP layer maps prompts to this SoccerTrack v2 BAS action set:

- `Pass`
- `Drive`
- `Shot`
- `Header`
- `High Pass`
- `Out`
- `Cross`
- `Throw In`
- `Ball Player Block`
- `Player Successful Tackle`
- `Free Kick`
- `Goal`

## Files

- `wan_model.py`: model components (VAE, T5 wrapper, DiT blocks, DDPM schedule, CFG sampler)
- `nlp_mapper.py`: maps raw prompts/captions to SoccerTrack v2 actions with LLM + rule fallback
- `train.py`: training loop with 80/20 train/validation split from `soccertrack_data/`
- `infer.py`: checkpoint-based sampling; can use explicit prompt or sample prompt from `soccertrack_data/`

## Training

```bash
python train.py \
  --data_dir ./soccertrack_data \
  --normalize_captions \
  --epochs 10 \
  --batch_size 2
```

Notes:

- Input videos are resized to the configured resolution and normalized to `[-1, 1]`.
- Validation runs every epoch on the 20% split.
- Checkpoints are saved to `checkpoints/`.
- With `--normalize_captions`, captions are mapped to SoccerTrack actions before T5 encoding.

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

By default, inference runs the prompt mapper. Disable it with:

```bash
python infer.py \
  --checkpoint checkpoints/wan_epoch_010.pt \
  --prompt "counter attack from the right wing" \
  --disable_prompt_mapper
```

## LLM API key (not committed)

Prompt mapping can call a hosted LLM API when an API key is set in env.

```bash
export OPENAI_API_KEY="your_key_here"
python infer.py --checkpoint checkpoints/wan_epoch_010.pt --prompt "quick pass then a shot"
```

If no key is present (or the API call fails), mapper falls back to local rule-based matching.

## Dependencies

Core:

- `torch`
- `torchvision` (video loading)
- `transformers` (T5 encoder)
- `Pillow`
- `imageio`
- `numpy`

CUDA is used automatically when available.
