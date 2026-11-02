import argparse
import sys
from pathlib import Path
from .pipeline import Wav2LipPipeline


def main():
    parser = argparse.ArgumentParser(description="Wav2Lip: Accurate lip-sync video generator")
    parser.add_argument("--checkpoint_path", type=str, default="gan", help="Model checkpoint ('gan', 'base', or path)")
    parser.add_argument("--face", type=str, required=True, help="Path to input face video or image")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--outfile", type=str, default="results/result.mp4", help="Output video path")
    parser.add_argument("--resize_factor", type=int, default=1, help="Downscale factor (1=original, 2=half resolution)")
    parser.add_argument("--pads", nargs=4, type=int, default=[0, 10, 0, 0], help="Padding: top bottom left right")
    parser.add_argument("--nosmooth", action="store_true", help="Disable temporal face smoothing")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS for static image input")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'mps', 'cpu')")

    args = parser.parse_args()

    print(f"[Wav2Lip] Initializing pipeline (checkpoint={args.checkpoint_path})...")
    pipeline = Wav2LipPipeline(checkpoint=args.checkpoint_path, device=args.device)

    print(f"[Wav2Lip] Generating video: face={args.face}, audio={args.audio}...")
    out = pipeline.generate(
        face=args.face,
        audio_path=args.audio,
        outfile=args.outfile,
        resize_factor=args.resize_factor,
        pads=args.pads,
        nosmooth=args.nosmooth,
        fps=args.fps,
    )
    print(f"[Wav2Lip] Video saved to: {out}")


if __name__ == "__main__":
    main()
