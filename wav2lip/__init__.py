"""
Wav2Lip: Accurately Lip-syncing Videos In The Wild
Packaged and modernized as a pip-installable Python package.
"""

from .pipeline import Wav2LipPipeline
from .models.wav2lip import Wav2Lip

__version__ = "0.2.0"
__all__ = ["Wav2LipPipeline", "Wav2Lip"]
