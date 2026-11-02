import os
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "camenduru/Wav2Lip"

def get_checkpoint_path(model_type: str = "gan") -> Path:
    """
    Returns the path to the Wav2Lip model checkpoint.
    If not already present in the local cache, automatically downloads it from Hugging Face.
    """
    filename = "wav2lip_gan.pth" if model_type == "gan" else "wav2lip.pth"
    # Check if local checkpoint exists in legacy checkpoints directory first
    local_default = Path(__file__).resolve().parent.parent / "checkpoints" / filename
    if local_default.exists() and local_default.stat().st_size > 100 * 1024 * 1024:
        return local_default

    cached_file = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"checkpoints/{filename}",
    )
    return Path(cached_file)


def get_face_detector_path() -> Path:
    """
    Returns the path to the S3FD face detector weights.
    If not present locally, automatically downloads it from Hugging Face.
    """
    local_default = (
        Path(__file__).resolve().parent
        / "face_detection"
        / "detection"
        / "sfd"
        / "s3fd.pth"
    )
    if local_default.exists() and local_default.stat().st_size > 50 * 1024 * 1024:
        return local_default

    cached_file = hf_hub_download(
        repo_id=REPO_ID,
        filename="face_detection/detection/sfd/s3fd.pth",
    )
    return Path(cached_file)
