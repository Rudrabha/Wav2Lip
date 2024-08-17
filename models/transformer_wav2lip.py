import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class TransformerWav2Lip(nn.Module):
    def __init__(self):
        super(TransformerWav2Lip, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),
            
            nn.Sequential(Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),
            
            nn.Sequential(Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),
            
            nn.Sequential(Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),
            
            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),
            
            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),
            
            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),
            
            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
                          Conv2d(512, 512, kernel_size=1, stride=1, padding=0))
        ])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
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

            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 1024, kernel_size=3, stride=1, padding=0),
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)
        )

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)),

            nn.Sequential(Conv2dTranspose(1536, 768, kernel_size=3, stride=1, padding=0),
                          Conv2d(768, 768, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2dTranspose(1280, 640, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(640, 640, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(640, 640, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2dTranspose(896, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2dTranspose(640, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2dTranspose(192, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2dTranspose(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True))
        ])
        
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=6)

        self.output_block = nn.Sequential(
            Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, audio_sequences, face_sequences):
        print("-----")
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())

        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences)
        face_features = []
        this_face_sequence = face_sequences

        for f in self.face_encoder_blocks:
            this_face_sequence = f(this_face_sequence)
            face_features.append(this_face_sequence)

        x = audio_embedding

        # Prepare data for transformer
        x = x.view(x.size(0), -1).unsqueeze(1)  # Adding an extra dimension for sequence length
        face_features = [f.view(f.size(0), -1).unsqueeze(1) for f in face_features]  # Adding an extra dimension

        x = torch.cat(face_features + [x], dim=1)  # Concatenate along the sequence length dimension

        x = x.view(x.size(0), -1, 1024)  # (B, sequence_length, d_model)

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        x = x.view(x.size(0), -1, 32, 32)  # Reshape back to 2D feature map

        for f in self.face_decoder_blocks:
            x = f(x)
            face_feature = face_features.pop().view(x.size(0), -1, x.size(2), x.size(3))  # Reshape back to original dimensions
            x = torch.cat((x, face_feature), dim=1)

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x, dim=2)
        else:
            outputs = x

        return outputs

