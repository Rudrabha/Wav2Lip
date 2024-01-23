import torch
import torch.onnx
from models.wav2lip import Wav2Lip
from sys import argv

model_name = "wav2lip"

model = Wav2Lip()

state = torch.load(f"checkpoints/{model_name}.pth", map_location=lambda storage, loc: storage)
cleaned_state = { key.replace('module.', ''): value for (key, value) in state['state_dict'].items() }

model.load_state_dict(cleaned_state)
model.eval()

# 16 samples (~200ms audio) of 80-mel spectrum data, transposed (0, 3, 1, 2)
# Autograd needed for...some reason? Maybe constant folding?
mel_batch = torch.randn(1, 1, 80, 16, requires_grad=True)

# frame of 96x96 video centered on an upright face in BGR format, concatenated
# on channel (innermost) axis with the same face but with the bottom half
# zeroed out (i.e. 6 total color channels), then transposed (0, 3, 1, 2)
img_batch = torch.randn(1, 6, 96, 96, requires_grad=True)

torch.onnx.export(
    model,
    (mel_batch, img_batch),
    f"{model_name}.onnx",
    export_params=True,
    do_constant_folding=True,
    input_names=['mel', 'vid'],
    output_names=['gen'],
    dynamic_axes={
        'mel': {0: 'batch'},
        'vid': {0: 'batch'},
        'gen': {0: 'batch'}
    }
)