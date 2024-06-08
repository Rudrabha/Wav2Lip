import torch
from torch import nn
from torch.nn import functional as F
from .conv import Conv2d

class TransformerSyncnet(nn.Module):
    def __init__(self, embed_size, num_heads, num_encoder_layers, num_classes=2, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        self.face_encoder = nn.Sequential(
            # Added by eddy
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3), #192x192
            nn.Dropout(p=0.3),
            
            # End added
            Conv2d(32, 32, kernel_size=3, stride=2, padding=1), #96x96
            Conv2d(32, 32, kernel_size=(7, 7), stride=1, padding=3, residual=True),
            Conv2d(32, 64, kernel_size=(7, 7), stride=1, padding=3, residual=True),

            Conv2d(64, 64, kernel_size=5, stride=(1, 2), padding=1), #94x47
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #94x47
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),#94x47

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 47x24
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), # 47x24
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), # 47x24
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), # 47x24

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 24x 12
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12

            Conv2d(256, 256, kernel_size=3, stride=2, padding=1), #12x6
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #12x6
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #12x6

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1), #6x3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0), #6x3
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),) #6x3

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=3, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)


        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout),
            num_layers=num_encoder_layers
        )
        self.fc1 = nn.Linear(embed_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, face_embedding, audio_embedding):
        # Adding a dummy dimension to match the transformer input requirements
        audio_embedding = audio_embedding.unsqueeze(1)
        face_embedding = face_embedding.unsqueeze(1)

        # Concatenate lip frames and audio features
        combined = torch.cat((face_embedding, audio_embedding), dim=1)
        
        # Pass through the Transformer encoder
        transformer_output = self.transformer_encoder(combined)
        
        # Pool the output (mean pooling)
        pooled_output = transformer_output.mean(dim=1)
        
        out = self.relu(self.fc1(pooled_output))
        out = self.fc2(out)
        return out


    

