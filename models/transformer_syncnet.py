import torch
from torch import nn
from torch.nn import functional as F

class TransformerClassifier(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, num_classes=2):
        super(TransformerClassifier, self).__init__()
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=embedding_dim*4,
            dropout=0.1,
            activation='relu'
        )
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, audio_embedding, face_embedding):
        # Adding a dummy dimension to match the transformer input requirements
        audio_embedding = audio_embedding.unsqueeze(1)
        face_embedding = face_embedding.unsqueeze(1)
        transformer_output = self.transformer(audio_embedding, face_embedding)
        transformer_output = transformer_output.mean(dim=1)
        out = self.fc(transformer_output)
        return out

    

