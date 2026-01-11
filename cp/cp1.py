"""
Convenience wrapper around the `EnvironmentalAudioPipeline` in `c/cl1.py`.
This script exposes a single helper that runs the full pipeline on an input
audio file and writes results to the requested output directory.
"""

import sys
from pathlib import Path
from typing import Optional
# Add project root to path to allow importing 'c' module
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import torch

from c.cl1 import EnvironmentalAudioPipeline


def process_audio(input_wav_path: str, output_dir: str = "./output", device: Optional[str] = None):
    """
    Run the full EnvironmentalAudioPipeline on the given WAV file.

    Args:
        input_wav_path: Path to the input audio file (mono or multichannel).
        output_dir: Directory where outputs (separated/enhanced audio, report) are written.
        device: Torch device string. If None, picks CUDA when available else CPU.

    Returns:
        The pipeline results dictionary produced by `EnvironmentalAudioPipeline.process`.
    """
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = EnvironmentalAudioPipeline(device=resolved_device)
    return pipeline.process(input_wav_path, output_dir=output_dir)


def _cli():
    parser = argparse.ArgumentParser(
        description="Run the EnvironmentalAudioPipeline on an input WAV file."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input WAV file.")
    parser.add_argument(
        "--outdir", "-o", default="./outputs/c", help="Directory to store pipeline outputs."
    )
    parser.add_argument(
        "--device",
        "-d",
        default=None,
        help="Torch device to use (e.g., 'cpu' or 'cuda'). Defaults to auto-detect.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.outdir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    process_audio(str(input_path), output_dir=str(output_dir), device=args.device)


if __name__ == "__main__":
    _cli()