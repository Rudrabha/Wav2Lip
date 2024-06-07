import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet

class EfficientSyncNet_color(nn.Module):
    def __init__(self):
        super(EfficientSyncNet_color, self).__init__()

        face_efficientnet = EfficientNet.from_pretrained('efficientnet-b4', in_channels=48)
        self.face_encoder = nn.Sequential(
            nn.Conv2d(15, 48, kernel_size=(7, 7), stride=2, padding=3, bias=False),
            *list(face_efficientnet.children())[1:-4],  # Exclude the final fully connected layer and average pool
            nn.AdaptiveAvgPool2d((1, 1))
        )
        print(self.face_encoder)
        self.fc_face = nn.Linear(face_efficientnet._fc.in_features, 1024)  # Adjust to desired embedding size
        self.freeze_layers(self.face_encoder, 1)  # Freeze layers up to layer3

        audio_efficientnet = EfficientNet.from_pretrained('efficientnet-b4', in_channels=64)
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=2, padding=3, bias=False),
            *list(audio_efficientnet.children())[1:-4],  # Exclude the final fully connected layer and average pool
        )
        self.fc_audio = nn.Linear(audio_efficientnet._fc.in_features, 1024)  # Adjust to desired embedding size
        self.freeze_layers(self.audio_encoder, 1)  # Freeze layers up to layer3

        

    def freeze_layers(self, model, num_layers):
        """
        Freeze the first `num_layers` blocks of the given EfficientNet model.
        """
        layers = list(model.children())
        print("num of layers", len(layers))
        count = 0

        for layer in layers:
            
            count += 1
            print("layer:", count)
            if count > num_layers:
                break
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, audio_sequences, face_sequences):
        # Forward pass through face encoder
        face_embedding = self.face_encoder(face_sequences)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)
        face_embedding = self.fc_face(face_embedding)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        # Forward pass through audio encoder
        audio_embedding = self.audio_encoder(audio_sequences)
        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        audio_embedding = self.fc_audio(audio_embedding)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)

        return audio_embedding, face_embedding