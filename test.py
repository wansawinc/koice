"""
Generate speech audio from text using the finetuned Kashmiri F5-TTS model.

Usage:
    python test.py "اسلام علیکم"
    python test.py --text "اسلام علیکم"
    python test.py --text "Hello world" --model training/model_final.safetensors
    python test.py --text "کشمیری زبان" --ref_audio dataset/wavs/audio0000001.wav
    python test.py --text "کشمیری زبان" --output my_output.wav

Arguments:
    text                     Text to synthesize (positional or --text)
    --model PATH             Path to model checkpoint (default: training/model_final.safetensors)
    --vocab PATH             Path to vocab file (default: training/vocab.txt)
    --ref_audio PATH         Reference audio for voice cloning (default: first wav in dataset/wavs/)
    --ref_text TEXT           Text spoken in reference audio (default: auto from metadata.csv)
    --output PATH            Output audio file path (default: output.wav)
    --speed FLOAT            Speed factor, 1.0 = normal (default: 1.0)

Examples:
    python test.py "نمستے دنیا"
    python test.py --text "کشمیر بہت خوبصورت ہے" --output kashmir.wav
    python test.py --text "Hello" --model training/ckpt_epoch_50.pt --ref_audio dataset/wavs/audio0000005.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).parent
DEFAULT_MODEL = PROJECT_ROOT / "training" / "model_final.safetensors"
DEFAULT_VOCAB = PROJECT_ROOT / "training" / "vocab.txt"
DATASET_DIR = PROJECT_ROOT / "dataset"
WAVS_DIR = DATASET_DIR / "wavs"
METADATA_CSV = DATASET_DIR / "metadata.csv"


def find_default_ref_audio():
    """Find a reference audio file and its text from the dataset."""
    ref_audio = None
    ref_text = ""

    # Try to get from metadata.csv
    if METADATA_CSV.exists():
        with open(METADATA_CSV, "r", encoding="utf-8") as f:
            f.readline()  # skip header
            line = f.readline().strip()
            if line:
                parts = line.split("|", 1)
                if len(parts) == 2:
                    audio_path = parts[0]
                    ref_text = parts[1]
                    # Resolve filename
                    audio_file = audio_path.replace("\\", "/").split("/")[-1]
                    candidate = WAVS_DIR / audio_file
                    if candidate.exists():
                        ref_audio = str(candidate)

    # Fallback: just pick first wav
    if ref_audio is None and WAVS_DIR.exists():
        wavs = sorted(WAVS_DIR.glob("*.wav"))
        if wavs:
            ref_audio = str(wavs[0])
            ref_text = ""

    return ref_audio, ref_text


def load_model(model_path: str, vocab_path: str, device: str):
    """Load the finetuned F5-TTS model."""
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer
    from safetensors.torch import load_file

    # Model config (must match training)
    MODEL_CONFIG = {
        "dim": 1024,
        "depth": 22,
        "heads": 16,
        "ff_mult": 2,
        "text_dim": 512,
        "conv_layers": 4,
    }
    MEL_SPEC_CONFIG = {
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mel_channels": 100,
        "target_sample_rate": 24000,
    }

    print(f"Loading vocab: {vocab_path}")
    vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")

    print(f"Building model (vocab_size={vocab_size})...")
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

    print(f"Loading weights: {model_path}")
    model_path = str(model_path)

    if model_path.endswith(".safetensors"):
        state = load_file(model_path)
        # Strip ema_model prefix if present
        cleaned = {}
        for k, v in state.items():
            if k.startswith("ema_model."):
                cleaned[k[len("ema_model."):]] = v
            else:
                cleaned[k] = v
        model.load_state_dict(cleaned, strict=False)
    elif model_path.endswith(".pt"):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    else:
        print(f"ERROR: Unsupported model format: {model_path}")
        sys.exit(1)

    model = model.to(device).eval()
    print(f"Model loaded on {device}\n")
    return model, MEL_SPEC_CONFIG


def generate_speech(model, text, ref_audio_path, ref_text, output_path, speed=1.0, device="cuda"):
    """Generate speech from text using the model."""
    from f5_tts.infer.utils_infer import (
        load_vocoder,
        infer_process,
    )

    print(f"Reference audio: {ref_audio_path}")
    print(f"Reference text:  {ref_text[:80]}{'...' if len(ref_text) > 80 else ''}")
    print(f"Generate text:   {text}")
    print(f"Speed:           {speed}")
    print(f"Output:          {output_path}")
    print()

    # Load vocoder
    print("Loading vocoder...")
    vocoder = load_vocoder()

    # Generate
    print("Generating speech...")
    audio, sr, _ = infer_process(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        gen_text=text,
        model_obj=model,
        vocoder=vocoder,
        speed=speed,
    )

    # Save output
    sf.write(output_path, audio, sr)
    duration = len(audio) / sr
    print(f"\nDone! Saved to: {output_path}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Sample rate: {sr} Hz")


def generate_speech_simple(model, text, ref_audio_path, ref_text, output_path, speed=1.0, device="cuda"):
    """
    Simpler generation path using f5-tts CLI internals.
    Fallback if infer_process is unavailable.
    """
    import torchaudio
    from f5_tts.infer.utils_infer import load_vocoder

    print(f"Reference audio: {ref_audio_path}")
    print(f"Reference text:  {ref_text[:80]}{'...' if len(ref_text) > 80 else ''}")
    print(f"Generate text:   {text}")
    print(f"Output:          {output_path}")
    print()

    # Load reference audio
    audio_ref, sr_ref = torchaudio.load(ref_audio_path)
    if sr_ref != 24000:
        resampler = torchaudio.transforms.Resample(sr_ref, 24000)
        audio_ref = resampler(audio_ref)

    # Load vocoder
    print("Loading vocoder...")
    vocoder = load_vocoder()

    # Prepare text
    gen_text = ref_text + " " + text

    # Duration estimation (rough: ref_duration + generated proportional to text length)
    ref_duration = audio_ref.shape[1] / 24000
    gen_duration = ref_duration + len(text) * 0.1 / speed  # rough estimate

    print("Generating speech...")
    with torch.inference_mode():
        generated, _ = model.sample(
            cond=audio_ref.to(device),
            text=[gen_text],
            duration=int(gen_duration * 24000 / 256),
            vocoder=vocoder,
        )

    # Extract only the generated part (after reference)
    ref_frames = audio_ref.shape[1]
    generated_audio = generated[ref_frames:]

    sf.write(output_path, generated_audio.cpu().numpy(), 24000)
    duration = len(generated_audio) / 24000
    print(f"\nDone! Saved to: {output_path}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Sample rate: 24000 Hz")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Kashmiri speech from text using finetuned F5-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py "اسلام علیکم"
  python test.py --text "کشمیری زبان" --output speech.wav
  python test.py --text "Hello" --model training/ckpt_epoch_50.pt
        """,
    )

    parser.add_argument("positional_text", nargs="?", default=None,
                        help="Text to synthesize (alternative to --text)")
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="Text to synthesize")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help=f"Model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--vocab", "-v", type=str, default=None,
                        help=f"Vocab path (default: {DEFAULT_VOCAB})")
    parser.add_argument("--ref_audio", "-r", type=str, default=None,
                        help="Reference audio for voice style")
    parser.add_argument("--ref_text", type=str, default=None,
                        help="Text spoken in reference audio")
    parser.add_argument("--output", "-o", type=str, default="output.wav",
                        help="Output file path (default: output.wav)")
    parser.add_argument("--speed", "-s", type=float, default=1.0,
                        help="Speed factor (default: 1.0)")

    args = parser.parse_args()

    # Resolve text (positional or --text)
    text = args.text or args.positional_text
    if not text:
        parser.print_help()
        print("\nERROR: No text provided. Use: python test.py \"your text here\"")
        sys.exit(1)

    # Resolve model path
    model_path = Path(args.model) if args.model else DEFAULT_MODEL
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        print(f"  Train first with: python train.py")
        print(f"  Or specify path:  python test.py --model path/to/model.pt \"text\"")
        sys.exit(1)

    # Resolve vocab path
    vocab_path = Path(args.vocab) if args.vocab else DEFAULT_VOCAB
    if not vocab_path.exists():
        print(f"ERROR: Vocab not found: {vocab_path}")
        sys.exit(1)

    # Resolve reference audio
    if args.ref_audio:
        ref_audio = args.ref_audio
        ref_text = args.ref_text or ""
    else:
        ref_audio, ref_text = find_default_ref_audio()
        if args.ref_text:
            ref_text = args.ref_text

    if not ref_audio or not Path(ref_audio).exists():
        print("ERROR: No reference audio found.")
        print("  Provide one with: --ref_audio path/to/ref.wav")
        sys.exit(1)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model, mel_config = load_model(str(model_path), str(vocab_path), device)

    # Generate
    try:
        generate_speech(model, text, ref_audio, ref_text, args.output, args.speed, device)
    except (ImportError, Exception) as e:
        print(f"  Primary generation failed: {e}")
        print(f"  Trying simple generation path...")
        try:
            generate_speech_simple(model, text, ref_audio, ref_text, args.output, args.speed, device)
        except Exception as e2:
            print(f"\nERROR: Generation failed: {e2}")
            print(f"\nTry using f5-tts CLI directly:")
            print(f'  f5-tts_infer-cli --model "{model_path}" --vocab "{vocab_path}" \\')
            print(f'    --ref_audio "{ref_audio}" --ref_text "{ref_text}" \\')
            print(f'    --gen_text "{text}"')
            sys.exit(1)


if __name__ == "__main__":
    main()
