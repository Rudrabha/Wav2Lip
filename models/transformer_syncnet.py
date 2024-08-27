import torch
from torch import nn
from torch.nn import functional as F
from .conv import Conv2d

class TransformerSyncnet(nn.Module):
    def __init__(self, num_heads, num_encoder_layers):
        super(TransformerSyncnet, self).__init__()

        self.face_encoder = nn.Sequential(
            # Added by eddy
            Conv2d(15, 32, kernel_size=3, stride=1, padding=1), #192x192
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            
            # End added
            Conv2d(32, 64, kernel_size=3, stride=2, padding=1), #96x96
            Conv2d(64, 64, kernel_size=(7, 7), stride=1, padding=3, residual=True),
            Conv2d(64, 64, kernel_size=(7, 7), stride=1, padding=3, residual=True),

            Conv2d(64, 128, kernel_size=5, stride=(1, 2), padding=1), #94x47
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #94x47
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #94x47

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 47x24
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 47x24
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 47x24

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # 24x 12
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12

            Conv2d(512, 768, kernel_size=3, stride=1, padding=1), # 24x 12
            Conv2d(768, 768, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12
            Conv2d(768, 768, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12

            Conv2d(768, 384, kernel_size=3, stride=2, padding=1), #12x6
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True), #12x6
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True), #12x6

            Conv2d(384, 128, kernel_size=3, stride=(2,1), padding=1), #6x6
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #12x6
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #12x6

            Conv2d(128, 64, kernel_size=3, stride=1, padding=1), #6x6
            
            )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 80x16
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 40x8
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 20x4
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1), # 10x4
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 10x4
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 256, kernel_size=3, stride=(2, 1), padding=1), # 5x4
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 128, kernel_size=3, stride=1, padding=1), # 5x4
            Conv2d(128, 64, kernel_size=3, stride=1, padding=1),)


        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1024, nhead=num_heads, dropout=0.),
            num_layers=num_encoder_layers
        )

        self.relu = nn.LeakyReLU(0.01, inplace=True)

        self.fc = nn.Linear(1152, 512) 
        self.fc2 = nn.Linear(1280, 512) 
        self.fc3 = nn.Linear(1024, 2) 
        

    def forward(self, face_embedding, audio_embedding):

        face_embedding = self.face_encoder(face_embedding)
        audio_embedding = self.audio_encoder(audio_embedding)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        face_embedding = self.fc(face_embedding)
        audio_embedding = self.fc2(audio_embedding)

        # normalise them
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        # Concatenate lip frames and audio features
        combined = torch.cat((face_embedding, audio_embedding), dim=1)
        
        # Make sure combined is 1-dimensional
        combined = combined.view(combined.size(0), -1)

        # Pass through the Transformer encoder, the input size is 1024
        transformer_output = self.transformer_encoder(combined)
        out = self.relu(transformer_output)
        out = self.fc3(out)
        
        return out, audio_embedding, face_embedding
