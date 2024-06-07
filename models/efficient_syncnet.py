import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet

class EfficientSyncNet_color(nn.Module):
    def __init__(self):
        super(EfficientSyncNet_color, self).__init__()

        # Modify EfficientNet for face encoding
        self.face_encoder = EfficientNet.from_pretrained('efficientnet-b7', in_channels=15)
        self.fc_face = nn.Linear(self.face_encoder._fc.in_features, 1024)  # Adjust to desired embedding size
        self.freeze_layers(self.face_encoder, 4)  # Freeze initial layers

        # Modify EfficientNet for audio encoding
        self.audio_encoder = EfficientNet.from_pretrained('efficientnet-b7', in_channels=1)
        self.fc_audio = nn.Linear(self.audio_encoder._fc.in_features, 1024)  # Adjust to desired embedding size
        self.freeze_layers(self.audio_encoder, 4)  # Freeze initial layers

    def freeze_layers(self, model, num_layers):
        """
        Freeze the first `num_layers` blocks of the given EfficientNet model.
        """
        layers = list(model.children())
        count = 0
        for layer in layers:
            count += 1
            if count > num_layers:
                break
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, audio_sequences, face_sequences):
        # Forward pass through face encoder
        face_embedding = self.face_encoder.extract_features(face_sequences)
        face_embedding = F.adaptive_avg_pool2d(face_embedding, (1, 1)).squeeze()
        face_embedding = self.fc_face(face_embedding)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        # Forward pass through audio encoder
        audio_embedding = self.audio_encoder.extract_features(audio_sequences)
        audio_embedding = F.adaptive_avg_pool2d(audio_embedding, (1, 1)).squeeze()
        audio_embedding = self.fc_audio(audio_embedding)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)

        return audio_embedding, face_embedding