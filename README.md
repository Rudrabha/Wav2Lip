# Wav2Lip Engine (pip-installable)

Modernized, pip-installable Python package for **Wav2Lip: Accurately Lip-syncing Videos In The Wild**.

This distribution provides a clean Python library API, automatic model downloading via Hugging Face Hub, and compatibility fixes for modern PyTorch (2.x), Librosa (0.10+), and Apple Silicon (MPS).

---

## Key Features

- **Direct Python API**: No CLI subprocesses required. Load the model once in-memory and generate lip-synced videos with `Wav2LipPipeline`.
- **Automatic Weight Caching**: Pre-trained weights (`wav2lip_gan.pth` and `s3fd.pth`) are automatically downloaded from Hugging Face Hub on first use to `~/.cache/huggingface/`.
- **Modern PyTorch 2.x Support**: Fixed `weights_only=False` unpickling behavior in PyTorch 2.4+.
- **Apple Silicon GPU Acceleration**: Native support for Apple Silicon Metal Performance Shaders (`mps`), alongside NVIDIA `cuda` and `cpu`.
- **Modern Audio Stack**: Fixed Librosa 0.10+ keyword arguments and replaced deprecated audio writers with `soundfile`.
- **Boundary Frame Protection**: Robust face detection that gracefully handles leading/trailing black frames, logos, and fade transitions without crashing.

---

## Installation

### From Git Repository:
```bash
pip install git+https://github.com/<your-username>/Wav2Lip.git
```

### Local Development Install:
```bash
git clone https://github.com/<your-username>/Wav2Lip.git
cd Wav2Lip
pip install -e .
```

---

## Python API Usage

```python
from wav2lip import Wav2LipPipeline

# 1. Initialize the pipeline (automatically downloads weights on first run)
# Options for checkpoint: 'gan' (default), 'base', or a local file path
pipeline = Wav2LipPipeline(checkpoint="gan")

# 2. Generate lip-synced video
output_path = pipeline.generate(
    face="speaker_face.mp4",        # Can be video or static image (.jpg, .png, etc.)
    audio_path="speech.wav",        # Synthesized speech audio
    outfile="lip_synced_output.mp4",# Destination path
    resize_factor=1,                # 1 = full resolution, 2 = 2x downscale (faster on CPU)
    pads=(0, 10, 0, 0),             # Chin padding (top, bottom, left, right)
    nosmooth=False,                 # Temporal smoothing for natural mouth transitions
)

print(f"Generated video saved to: {output_path}")
```

---

## CLI Usage

You can also use the packaged CLI tool directly from the terminal:

```bash
wav2lip --face input_video.mp4 --audio speech.wav --outfile output.mp4 --pads 0 10 0 0
```

---

## System Requirements

- **Python**: 3.9 or higher
- **FFmpeg**: Required for audio/video muxing (`sudo apt install ffmpeg` on Ubuntu/Debian, `brew install ffmpeg` on macOS)
- **PyTorch**: 2.0+

---

## License

This package builds on the original Wav2Lip paper by Rudrabha Mukhopadhyay, Vinay P. Namboodiri, and C.V. Jawahar. The original research code is licensed under CC BY-NC-SA 4.0.
