import torch
from torch import nn
from torch.nn import functional as F
from .conv import Conv2d

class TransformerSyncnet(nn.Module):
    def __init__(self, embed_size, num_heads, num_encoder_layers, num_classes=2, dropout=0.1):
        super(TransformerSyncnet, self).__init__()

        self.face_encoder = nn.Sequential(
            # Added by eddy
            Conv2d(15, 128, kernel_size=(7, 7), stride=1, padding=3), #192x192
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            # End added
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1), #96x96
            Conv2d(128, 128, kernel_size=(7, 7), stride=1, padding=3, residual=True),
            Conv2d(128, 128, kernel_size=(7, 7), stride=1, padding=3, residual=True),

            Conv2d(128, 256, kernel_size=5, stride=(1, 2), padding=1), #94x47
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #94x47
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),#94x47

            Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # 47x24
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 47x24
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 47x24

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # 24x 12
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1), #12x6
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), #12x6
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), #12x6

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1), #6x3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0), #6x3
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),) #6x3

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 256, kernel_size=3, stride=3, padding=1),
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

        self.fc1 = nn.Linear(1536, 1024)
        self.fc4 = nn.Linear(1024, embed_size)
        self.fc2 = nn.Linear(embed_size, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, face_embedding, audio_embedding):

        face_embedding = self.face_encoder(face_embedding)
        audio_embedding = self.audio_encoder(audio_embedding)
        #lip_landmark = lip_landmark.flatten()  # (180,3) shape becomes (540,)


        # Adding a dummy dimension to match the transformer input requirements
        audio_embedding = audio_embedding.unsqueeze(1)
        face_embedding = face_embedding.unsqueeze(1)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        # normalise them
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        # Concatenate lip frames and audio features
        combined = torch.cat((face_embedding, audio_embedding), dim=1)
        
        
        # Make sure combined is 1-dimensional
        combined = combined.view(combined.size(0), -1)
        
        #lip_landmark = lip_landmark.view(5, -1)  # Flatten bbb to shape [5, 540]
        #combined = torch.cat((combined, lip_landmark), dim=1)  # Concatenate along the last dimension

        # The embedding size is 1024, fc1 will reduce it to 256
        combined = self.fc1(combined)
        combined = self.fc4(combined)
        combined = self.relu(combined)
        
        # Pass through the Transformer encoder
        transformer_output = self.transformer_encoder(combined)
        out = self.relu(transformer_output)

        # The output from the transformer should be 256, add a dropout layer to drop 0.2, then further reduce to 128 classes
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        
        # Further reduce to 2 classes
        out = self.fc3(out)
        
        return out
