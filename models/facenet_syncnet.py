import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from facenet_pytorch import InceptionResnetV1

from .conv import Conv2d

class ResSyncNet_color(nn.Module):
    def __init__(self):
        super(ResSyncNet_color, self).__init__()

        # Initialize FaceNet model
        facenet = InceptionResnetV1(pretrained='vggface2', classify=False)

        # Modify the output layer to match the desired output size
        facenet.last_linear = nn.Linear(facenet.last_linear.in_features, 1024)


        self.face_encoder = nn.Sequential(
            nn.Conv2d(15, 64, kernel_size=(7, 7), stride=2, padding=3, bias=False),
            facenet
        )
        
        self.freeze_layers(self.face_encoder, '3')  # Freeze layers up to layer3


        # Modify ResNet50 for audio encoding
        resnet_audio = models.resnet152(pretrained=True)
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=2, padding=3, bias=False),
            *list(resnet_audio.children())[1:-2],  # Exclude the final fully connected layer and average pool
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc_audio = nn.Linear(resnet_audio.fc.in_features, 1024)  # Adjust to desired embedding size
        self.freeze_layers(self.audio_encoder, '3')  # Freeze layers up to layer3


    def freeze_layers(self, model, freeze_until):
        """
        Freeze the layers of the given model up to the specified layer.
        """
        freeze = True
        for name, child in model.named_children():
            print("layer name", name)
            if name == freeze_until:
                freeze = False
            if freeze:
                for param in child.parameters():
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
