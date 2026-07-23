"""
F5-TTS Finetuning for Kashmiri TTS

Single-file training script that:
1. Downloads F5-TTS v1 Base pretrained checkpoint + vocab
2. Builds a custom vocab covering Kashmiri (Nastaliq) characters from your dataset
3. Loads dataset from dataset/metadata.csv
4. Finetunes the model
5. Saves checkpoints to training/ folder

Requirements:
pip install f5-tts torch torchaudio cached-path safetensors omegaconf soundfile numpy    


Usage:
    python train.py                          # Start training
    python train.py --resume training/ckpt_epoch_3.pt  # Resume from checkpoint
    python train.py --epochs 10 --batch-size 4 --lr 1e-5

Output:
    training/
        ckpt_epoch_1.pt      - Checkpoint after epoch 1
        ckpt_epoch_2.pt      - Checkpoint after epoch 2
        ...
        model_final.safetensors - Final model (pruned, for inference)
        vocab.txt            - Vocabulary file (for inference)
        train.log            - Training log
"""

import argparse
import json
import math
import os
import sys
import time
import types
from pathlib import Path

# ============================================================
# Stub out HuggingFace datasets to avoid pyarrow dependency
# F5-TTS's dataset.py imports it but we don't need it
# ============================================================
_datasets_stub = types.ModuleType("datasets")
_datasets_stub.Dataset = type("Dataset", (), {})
_datasets_stub.load_from_disk = lambda *a, **kw: (_ for _ in ()).throw(
    NotImplementedError("stub")
)
sys.modules.setdefault("datasets", _datasets_stub)

import torch
import numpy as np
from cached_path import cached_path
from safetensors.torch import load_file, save_file

# torchaudio may not work on all platforms (Kaggle CUDA mismatch)
# Use soundfile as fallback for audio loading
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except (OSError, ImportError):
    TORCHAUDIO_AVAILABLE = False
    import soundfile as sf
    print("WARNING: torchaudio unavailable, using soundfile fallback")

    # Create a comprehensive torchaudio stub so f5_tts internals work
    import types as _types

    _torchaudio_stub = _types.ModuleType("torchaudio")

    def _sf_load(filepath, **kwargs):
        """soundfile-based replacement for torchaudio.load"""
        data, sample_rate = sf.read(filepath, dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T  # (channels, samples)
        return waveform, sample_rate

    def _sf_info(filepath):
        """soundfile-based replacement for torchaudio.info"""
        info = sf.info(filepath)
        _info = _types.SimpleNamespace(
            num_frames=info.frames,
            sample_rate=info.samplerate,
        )
        return _info

    _torchaudio_stub.load = _sf_load
    _torchaudio_stub.info = _sf_info

    # Stub transforms with actual MelSpectrogram using torch
    _transforms = _types.ModuleType("torchaudio.transforms")

    class _Resample(torch.nn.Module):
        def __init__(self, orig_freq, new_freq):
            super().__init__()
            self.orig_freq = orig_freq
            self.new_freq = new_freq
        def forward(self, waveform):
            ratio = self.new_freq / self.orig_freq
            new_length = int(waveform.shape[-1] * ratio)
            return torch.nn.functional.interpolate(
                waveform.unsqueeze(0), size=new_length, mode="linear", align_corners=False
            ).squeeze(0)

    class _MelSpectrogram(torch.nn.Module):
        """Pure PyTorch MelSpectrogram (no torchaudio dependency)."""
        def __init__(self, sample_rate=24000, n_fft=1024, win_length=1024,
                     hop_length=256, n_mels=100, power=1, center=True,
                     pad_mode="reflect", norm="slaney", mel_scale="slaney", **kwargs):
            super().__init__()
            self.n_fft = n_fft
            self.win_length = win_length or n_fft
            self.hop_length = hop_length
            self.center = center
            self.pad_mode = pad_mode
            self.power = power

            # Create mel filterbank
            mel_fb = self._create_mel_filterbank(sample_rate, n_fft, n_mels, norm, mel_scale)
            self.register_buffer("mel_fb", mel_fb)

            window = torch.hann_window(self.win_length)
            self.register_buffer("window", window)

        def _create_mel_filterbank(self, sr, n_fft, n_mels, norm, mel_scale):
            """Create mel filterbank matrix."""
            def hz_to_mel(hz):
                if mel_scale == "slaney":
                    if hz < 1000:
                        return 3.0 * hz / 200.0
                    else:
                        return 15.0 + 27.0 * math.log(hz / 1000.0) / math.log(6.4)
                return 2595.0 * math.log10(1.0 + hz / 700.0)

            def mel_to_hz(mel):
                if mel_scale == "slaney":
                    if mel < 15.0:
                        return 200.0 * mel / 3.0
                    else:
                        return 1000.0 * math.exp((mel - 15.0) * math.log(6.4) / 27.0)
                return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

            import math

            f_min = 0.0
            f_max = sr / 2.0
            mel_min = hz_to_mel(f_min)
            mel_max = hz_to_mel(f_max)

            mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
            hz_points = torch.tensor([mel_to_hz(m.item()) for m in mel_points])
            bin_points = torch.floor((n_fft + 1) * hz_points / sr).long()

            n_freqs = n_fft // 2 + 1
            fb = torch.zeros(n_freqs, n_mels)

            for i in range(n_mels):
                left = bin_points[i]
                center = bin_points[i + 1]
                right = bin_points[i + 2]

                for j in range(left, center):
                    if center != left:
                        fb[j, i] = (j - left) / (center - left)
                for j in range(center, right):
                    if right != center:
                        fb[j, i] = (right - j) / (right - center)

            if norm == "slaney":
                enorm = 2.0 / (hz_points[2:n_mels+2] - hz_points[:n_mels])
                fb *= enorm.unsqueeze(0)

            return fb  # (n_freqs, n_mels)

        def forward(self, waveform):
            if self.center:
                pad_amount = self.n_fft // 2
                waveform = torch.nn.functional.pad(waveform, (pad_amount, pad_amount), mode=self.pad_mode)

            # STFT
            spec = torch.stft(
                waveform.squeeze(0) if waveform.dim() > 1 else waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window.to(waveform.device),
                return_complex=True,
            )
            # Power/magnitude spectrogram
            if self.power == 1:
                spec = spec.abs()
            else:
                spec = spec.abs().pow(self.power)

            # Apply mel filterbank
            mel_fb = self.mel_fb.to(spec.device)
            if spec.dim() == 2:
                mel_spec = torch.matmul(spec.T, mel_fb).T  # (n_mels, time)
            else:
                mel_spec = torch.matmul(spec.transpose(-1, -2), mel_fb).transpose(-1, -2)

            return mel_spec

    _transforms.Resample = _Resample
    _transforms.MelSpectrogram = _MelSpectrogram
    _torchaudio_stub.transforms = _transforms

    # Stub functional
    _functional = _types.ModuleType("torchaudio.functional")
    _torchaudio_stub.functional = _functional

    sys.modules["torchaudio"] = _torchaudio_stub
    sys.modules["torchaudio.transforms"] = _transforms
    sys.modules["torchaudio.functional"] = _functional
    torchaudio = _torchaudio_stub

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAINING_DIR = PROJECT_ROOT / "training"
METADATA_CSV = DATASET_DIR / "metadata.csv"

# F5-TTS v1 Base model (publicly available)
PRETRAINED_CKPT = "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
PRETRAINED_VOCAB = "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt"

# Model architecture (must match F5TTS_v1_Base)
MODEL_CONFIG = {
    "dim": 1024,
    "depth": 22,
    "heads": 16,
    "ff_mult": 2,
    "text_dim": 512,
    "conv_layers": 4,
}

# Mel spectrogram settings (must match pretrained)
MEL_SPEC_CONFIG = {
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 100,
    "target_sample_rate": 24000,
}

# Training defaults
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 4  # Adjust based on your GPU VRAM
DEFAULT_LR = 7.5e-5
DEFAULT_WARMUP_STEPS = 200
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SAVE_EVERY_N_EPOCHS = 20


# ============================================================
# VOCAB BUILDER
# ============================================================
def build_kashmiri_vocab(metadata_csv: Path, output_path: Path):
    """
    Start from the pretrained F5-TTS vocab (preserving exact order and entries),
    then append any new Kashmiri characters not already present.
    This ensures the text_embed layer stays compatible with pretrained weights.
    """
    # Load original vocab exactly as-is (preserves order + multi-char tokens)
    orig_vocab_path = str(cached_path(PRETRAINED_VOCAB))
    original_lines = []
    original_chars = set()
    with open(orig_vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = line.strip()
            if entry:
                original_lines.append(entry)
                original_chars.add(entry)

    print(f"  Original vocab size: {len(original_lines)}")

    # Collect Kashmiri characters from dataset
    kashmiri_chars = set()
    with open(metadata_csv, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("|", 1)
            if len(parts) == 2:
                text = parts[1]
                kashmiri_chars.update(text)

    # Find new characters not in original vocab
    new_chars = sorted(kashmiri_chars - original_chars)
    print(f"  New Kashmiri characters to add: {len(new_chars)}")

    # Write: original vocab first (unchanged), then new chars appended
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in original_lines:
            f.write(entry + "\n")
        for ch in new_chars:
            f.write(ch + "\n")

    total_size = len(original_lines) + len(new_chars)
    print(f"  Final vocab size: {total_size} characters")
    print(f"  Saved to: {output_path}")
    return output_path


# ============================================================
# DATASET
# ============================================================
class KashmiriTTSDataset(torch.utils.data.Dataset):
    """
    Simple dataset that loads audio + text pairs from metadata.csv.
    Computes mel spectrograms on-the-fly.
    """

    def __init__(self, metadata_csv: Path, mel_spec_config: dict):
        self.entries = []
        self.target_sr = mel_spec_config["target_sample_rate"]
        self.hop_length = mel_spec_config["hop_length"]

        # Resolve wavs dir relative to metadata.csv location
        wavs_dir = metadata_csv.parent / "wavs"

        if not wavs_dir.exists():
            print(f"  WARNING: wavs/ dir not found at: {wavs_dir}")
            print(f"  Looking for alternatives...")
            # Try common alternatives
            alternatives = [
                metadata_csv.parent.parent / "wavs",  # one level up
                metadata_csv.parent.parent / "dataset" / "wavs",
                metadata_csv.parent.parent / "raw" / "segments",  # segments folder
            ]
            for alt in alternatives:
                if alt.exists():
                    wavs_dir = alt
                    print(f"  Found wavs at: {wavs_dir}")
                    break
            else:
                print(f"  ERROR: Cannot find wavs directory!")
                print(f"  Expected at: {metadata_csv.parent / 'wavs'}")

        print(f"  Wavs directory: {wavs_dir}")
        print(f"  Wavs dir exists: {wavs_dir.exists()}")
        if wavs_dir.exists():
            wav_count = len(list(wavs_dir.glob("*.wav")))
            print(f"  WAV files found: {wav_count}")

        with open(metadata_csv, "r", encoding="utf-8") as f:
            header = f.readline()  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|", 1)
                if len(parts) == 2:
                    audio_path, text = parts
                    # Handle both absolute and relative paths (cross-platform)
                    # Extract filename regardless of Windows/Linux path separators
                    audio_file = audio_path.replace("\\", "/").split("/")[-1]
                    resolved_path = wavs_dir / audio_file
                    if resolved_path.exists() and text.strip():
                        self.entries.append({
                            "audio_path": str(resolved_path),
                            "text": text.strip(),
                        })

        print(f"  Loaded {len(self.entries)} valid entries from dataset")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        return {
            "audio_path": entry["audio_path"],
            "text": entry["text"],
        }

    def get_frame_len(self, idx):
        """Approximate frame length for batching."""
        # Load audio to get duration
        info = torchaudio.info(self.entries[idx]["audio_path"])
        duration = info.num_frames / info.sample_rate
        return int(duration * self.target_sr / self.hop_length)


# ============================================================
# MODEL LOADING
# ============================================================
def download_and_load_model(vocab_path: Path, device: str):
    """Download pretrained F5-TTS v1 Base and load with custom vocab."""
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer

    print("\n[1/3] Loading vocabulary...")
    vocab_char_map, vocab_size = get_tokenizer(str(vocab_path), "custom")
    print(f"  Vocab size: {vocab_size}")

    print("\n[2/3] Building model architecture...")
    transformer = DiT(
        **MODEL_CONFIG,
        text_num_embeds=vocab_size,
        mel_dim=MEL_SPEC_CONFIG["n_mel_channels"],
    )
    model = CFM(
        transformer=transformer,
        mel_spec_kwargs=MEL_SPEC_CONFIG,
        vocab_char_map=vocab_char_map,
    )

    print("\n[3/3] Downloading pretrained checkpoint...")
    ckpt_path = str(cached_path(PRETRAINED_CKPT))
    print(f"  Checkpoint: {ckpt_path}")

    state = load_file(ckpt_path)

    # The published checkpoint has EMA weights prefixed with 'ema_model.'
    cleaned = {}
    skipped = 0
    for k, v in state.items():
        if k.startswith("ema_model."):
            cleaned[k[len("ema_model."):]] = v
        else:
            skipped += 1

    # Handle text_embed size mismatch: if our vocab is larger than pretrained,
    # copy pretrained weights into the first N rows and randomly init the rest.
    # The model's actual embedding size may differ from vocab_size (DiT adds +1 internally)
    embed_key = "transformer.text_embed.text_embed.weight"
    if embed_key in cleaned:
        pretrained_embed = cleaned[embed_key]  # (pretrained_vocab_size, embed_dim)
        pretrained_size = pretrained_embed.shape[0]
        # Get the actual size the model expects
        model_embed_size = model.state_dict()[embed_key].shape[0]
        if model_embed_size != pretrained_size:
            print(f"  Resizing text embedding: {pretrained_size} -> {model_embed_size}")
            # Initialize new rows with the mean of existing embeddings (stable, no NaN)
            mean_embed = pretrained_embed.mean(dim=0, keepdim=True)
            new_embed = mean_embed.expand(model_embed_size, -1).clone()
            # Copy pretrained weights for the overlapping portion
            copy_size = min(pretrained_size, model_embed_size)
            new_embed[:copy_size] = pretrained_embed[:copy_size]
            cleaned[embed_key] = new_embed

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"  Loaded {len(cleaned)} tensors (skipped {skipped} bookkeeping)")
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    return model.to(device), vocab_char_map


# ============================================================
# TRAINING LOOP
# ============================================================
def train(args):
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    # Detect multi-GPU and use accelerate if available
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_accelerate = False
    accelerator = None

    if num_gpus > 1:
        try:
            from accelerate import Accelerator
            accelerator = Accelerator(mixed_precision="fp16")
            use_accelerate = True
            device = accelerator.device
            print(f"Device: {device} (using accelerate with {num_gpus} GPUs)")
            for i in range(num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                      f"({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")
        except ImportError:
            print(f"WARNING: {num_gpus} GPUs detected but 'accelerate' not installed.")
            print(f"  Install with: pip install accelerate")
            print(f"  Falling back to single GPU.")
            device = "cuda"
            print(f"Device: {device}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        if device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Build vocab from dataset
    print("\n" + "=" * 60)
    print("STEP 1: Building vocabulary from dataset")
    print("=" * 60)
    vocab_path = TRAINING_DIR / "vocab.txt"
    build_kashmiri_vocab(METADATA_CSV, vocab_path)

    # Load model
    print("\n" + "=" * 60)
    print("STEP 2: Loading pretrained F5-TTS v1 Base model")
    print("=" * 60)
    model, vocab_char_map = download_and_load_model(vocab_path, device)
    model.train()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params / 1e6:.1f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.1f}M")

    # Load dataset
    print("\n" + "=" * 60)
    print("STEP 3: Loading dataset")
    print("=" * 60)
    dataset = KashmiriTTSDataset(METADATA_CSV, MEL_SPEC_CONFIG)

    if len(dataset) == 0:
        print("ERROR: No valid entries found in dataset!")
        sys.exit(1)

    # Try to use F5-TTS's built-in dataset and collate for proper mel handling
    try:
        from f5_tts.model.dataset import CustomDataset, collate_fn as f5_collate_fn

        # Get durations - handle different torchaudio versions / fallback to soundfile
        def get_audio_duration(path):
            if TORCHAUDIO_AVAILABLE:
                try:
                    info = torchaudio.info(path)
                    return info.num_frames / info.sample_rate
                except (AttributeError, RuntimeError):
                    waveform, sr = torchaudio.load(path)
                    return waveform.shape[1] / sr
            else:
                data, sr = sf.read(path)
                return len(data) / sr

        # Build rows in the format CustomDataset expects
        rows = []
        durations = []
        for entry in dataset.entries:
            dur = get_audio_duration(entry["audio_path"])
            rows.append({
                "audio_path": entry["audio_path"],
                "text": entry["text"],
                "duration": dur,
            })
            durations.append(dur)

        train_ds = CustomDataset(
            rows,
            durations=durations,
            **MEL_SPEC_CONFIG,
        )
        use_f5_dataset = True
        print(f"  Using F5-TTS CustomDataset (built-in mel computation)")
    except Exception as e:
        print(f"  ERROR: Could not initialize F5-TTS CustomDataset: {e}")
        print(f"  Make sure f5-tts is installed: pip install f5-tts")
        print(f"  And torchaudio is compatible: pip install torchaudio")
        sys.exit(1)

    # DataLoader
    is_cuda = (device == "cuda") if isinstance(device, str) else (str(device).startswith("cuda"))
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=f5_collate_fn,
        num_workers=0,  # Avoid multiprocessing issues on Windows
        pin_memory=is_cuda,
    )

    total_updates = args.epochs * len(loader)
    print(f"\n  Dataset size: {len(train_ds)}")
    print(f"  Batches per epoch: {len(loader)}")
    print(f"  Total training steps: {total_updates}")

    # Optimizer + Scheduler
    print("\n" + "=" * 60)
    print("STEP 4: Setting up optimizer")
    print("=" * 60)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Linear warmup then constant LR (good for finetuning)
    from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR

    warmup_steps = min(args.warmup_steps, total_updates // 4)
    warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    constant = ConstantLR(optimizer, factor=1.0, total_iters=total_updates - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, constant], milestones=[warmup_steps])

    print(f"  Learning rate: {args.lr}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Optimizer: AdamW (weight_decay=0.01)")

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"\n  Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        print(f"  Resumed at epoch={start_epoch}, step={global_step}")

    # Training log
    log_path = TRAINING_DIR / "train.log"
    log_file = open(log_path, "a", encoding="utf-8")

    # Mixed precision for faster training
    # When using accelerate, it handles AMP internally
    if use_accelerate:
        model, optimizer, loader, scheduler = accelerator.prepare(
            model, optimizer, loader, scheduler
        )
        use_amp = False  # accelerate handles this
        scaler = None
        print(f"  Multi-GPU: accelerate prepared (fp16 handled internally)")
    else:
        use_amp = (device == "cuda") if isinstance(device, str) else (str(device) == "cuda")
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print("\n" + "=" * 60)
    print("STEP 5: Starting training")
    print("=" * 60)
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Save every: {args.save_every} epochs")
    print(f"  AMP (mixed precision): {use_amp}")
    print()

    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_losses = []

        for batch_idx, batch in enumerate(loader):
            # Debug: print batch info on first batch of first epoch
            if epoch == start_epoch and batch_idx == 0:
                print(f"  [DEBUG] Batch keys: {list(batch.keys()) if isinstance(batch, dict) else type(batch)}")
                if isinstance(batch, dict):
                    for k, v in batch.items():
                        if hasattr(v, 'shape'):
                            print(f"  [DEBUG]   {k}: shape={v.shape}, dtype={v.dtype}")
                        elif isinstance(v, list):
                            print(f"  [DEBUG]   {k}: list len={len(v)}")
                        else:
                            print(f"  [DEBUG]   {k}: {type(v)}")

            # F5-TTS CustomDataset returns mel + text in batch
            if use_accelerate:
                mel = batch["mel"]
                mel_lengths = batch["mel_lengths"]
            else:
                mel = batch["mel"].to(device)
                mel_lengths = batch["mel_lengths"].to(device)
            text = batch["text"]

            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss, _, _ = model(mel.permute(0, 2, 1), text=text, lens=mel_lengths)
            else:
                loss, _, _ = model(mel.permute(0, 2, 1), text=text, lens=mel_lengths)

            # Skip NaN losses (can happen in early steps)
            if torch.isnan(loss) or torch.isinf(loss):
                if batch_idx < 5:
                    print(f"  [WARN] NaN/Inf loss at step {global_step+1}, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)

            if use_accelerate:
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            elif use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()
            global_step += 1
            epoch_losses.append(loss.item())

            # Log every 10 steps
            if global_step % 10 == 0 or batch_idx == 0:
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                avg_loss = np.mean(epoch_losses[-50:])  # running avg of last 50
                line = (
                    f"epoch={epoch+1}/{args.epochs} "
                    f"step={global_step}/{total_updates} "
                    f"loss={loss.item():.4f} "
                    f"avg_loss={avg_loss:.4f} "
                    f"lr={lr_now:.2e} "
                    f"elapsed={elapsed:.0f}s"
                )
                print(line)
                log_file.write(line + "\n")
                log_file.flush()

        # End of epoch
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
        elapsed = time.time() - t0
        epoch_line = (
            f"\n>>> Epoch {epoch+1}/{args.epochs} complete | "
            f"avg_loss={avg_epoch_loss:.4f} | "
            f"elapsed={elapsed:.0f}s\n"
        )
        print(epoch_line)
        log_file.write(epoch_line + "\n")
        log_file.flush()

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = TRAINING_DIR / f"ckpt_epoch_{epoch+1}.pt"
            # Unwrap model if using accelerate (DDP wrapper)
            save_model = accelerator.unwrap_model(model) if use_accelerate else model
            ckpt_data = {
                "step": global_step,
                "epoch": epoch + 1,
                "model_state_dict": save_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_epoch_loss,
                "config": MODEL_CONFIG,
            }
            torch.save(ckpt_data, ckpt_path)
            # Verify checkpoint is readable
            try:
                torch.load(ckpt_path, map_location="cpu", weights_only=False)
                print(f"  Saved checkpoint: {ckpt_path} (verified)")
            except Exception as e:
                print(f"  WARNING: Checkpoint may be corrupt: {e}")
                # Try saving again
                torch.save(ckpt_data, ckpt_path)
                print(f"  Re-saved checkpoint: {ckpt_path}")

        # Track best
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss

    log_file.close()

    # Save final model in inference-ready format (safetensors, EMA-style)
    print("\n" + "=" * 60)
    print("STEP 6: Saving final model for inference")
    print("=" * 60)

    # Unwrap model if using accelerate
    save_model = accelerator.unwrap_model(model) if use_accelerate else model

    # Save as safetensors with ema_model prefix (what F5-TTS expects for inference)
    final_state = {}
    for k, v in save_model.state_dict().items():
        final_state[f"ema_model.{k}"] = v

    final_path = TRAINING_DIR / "model_final.safetensors"
    save_file(final_state, str(final_path))
    print(f"  Model saved: {final_path}")
    print(f"  Vocab saved: {vocab_path}")
    print(f"  Best loss: {best_loss:.4f}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nTo use this model for inference with F5-TTS:")
    print(f"  Model: {final_path}")
    print(f"  Vocab: {vocab_path}")
    print(f"\nExample inference command:")
    print(f'  f5-tts_infer-cli --model "{final_path}" --vocab "{vocab_path}" \\')
    print(f'    --ref_audio "path/to/reference.wav" --ref_text "reference text" \\')
    print(f'    --gen_text "text to synthesize"')


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="F5-TTS Finetuning for Kashmiri")

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS,
                        help=f"Warmup steps (default: {DEFAULT_WARMUP_STEPS})")
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP,
                        help=f"Gradient clipping norm (default: {DEFAULT_GRAD_CLIP})")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY_N_EPOCHS,
                        help=f"Save checkpoint every N epochs (default: {DEFAULT_SAVE_EVERY_N_EPOCHS})")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
