import torch
from torch import nn
from torch.nn import functional as F
import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class TransformerWav2Lip(nn.Module):
    def __init__(self, num_heads, num_encoder_layers):
        super(TransformerSyncnet, self).__init__()
        '''
        Outpu = Input + (k-1) x S

        Where:
        Input is the receptive field size from the previous layer.
        k is the kernel size.
        S is the stride.
        '''
        self.face_encoder = nn.Sequential(
            
            Conv2d(15, 32, kernel_size=7, stride=1, padding=3), #192x192, 1+(7−1)×1=7
            
            Conv2d(32, 64, kernel_size=7, stride=2, padding=3), #96x96, 7+(7−1)×2=7+12=19
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), # 19+(3−1)×1=19+2=21
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), # 21+(3−1)×1=19+2=23

            Conv2d(64, 128, kernel_size=5, stride=(1, 2), padding=1), #94x47, 23+(5−1)×1=19+2=23
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #94x47
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #94x47

            Conv2d(128, 256, kernel_size=5, stride=2, padding=2), # 47x24
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 47x24
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 47x24

            Conv2d(256, 256, kernel_size=5, stride=2, padding=2), # 24x 12
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12

            Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 24x 12
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12

            Conv2d(512, 256, kernel_size=5, stride=2, padding=2), #12x6
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #12x6
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #12x6

            Conv2d(256, 128, kernel_size=3, stride=(2,1), padding=1), #6x6
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #12x6
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #12x6

            Conv2d(128, 64, kernel_size=3, stride=1, padding=1), #6x6
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1), #6x6

            Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # The receptive field up to here is 99x99
            
            )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 80x16
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)


        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=num_heads, dropout=0.),
            num_layers=num_encoder_layers
        )

        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.fc1 = nn.Linear(384, 256) 
        self.fc2 = nn.Linear(512, 256) 

        self.output_block = nn.Sequential(Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())
        

    def forward(self, face_embedding, audio_embedding):

        face_embedding = self.face_encoder(face_embedding)
        audio_embedding = self.audio_encoder(audio_embedding)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        face_embedding = self.fc1(face_embedding)
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
        
        output = self.output_block(transformer_output)
        
        batch_size = transformer_output.size(0)
        transformer_output = transformer_output.view(batch_size, 96, 96, 192)  # Example shape, adjust as needed

        output = self.output_block(transformer_output)
        
        return output


