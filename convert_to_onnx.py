import torch
import torch.onnx
from models import Wav2Lip

model = Wav2Lip()
checkpoint_path = "checkpoints/wav2lip.pth"
modelz = torch.load(checkpoint_path, map_location=torch.device('cpu'))

device="cpu"
s = modelz["state_dict"]
new_s = {}
for k, v in s.items():
    new_s[k.replace('module.', '')] = v

model.load_state_dict(new_s)
model = model.to(device)

model.eval()
input_shape1 = (8, 1, 80, 16)
input_shape2 = (8, 6, 96, 96)

torch.onnx.export(model,
                  (torch.randn(*input_shape1, device=device), torch.randn(*input_shape2, device=device)),
                  "wav2lip.onnx",
                  input_names=["mel_spectrogram", "video_frames"],
                  output_names=["predicted_frames"],
                  verbose=True,
                  opset_version=18)
