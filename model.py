import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch

class StreamingTransformer(nn.Module):
    def __init__(self, chunk_size=512, num_classes=21, d_model=128, nhead=8, num_layers=6):
        super(StreamingTransformer, self).__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model
        
        self.embedding = nn.Linear(3 * chunk_size, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes * chunk_size)
        
    def forward(self, x, state=None):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the input
        x = self.embedding(x)
        if state is None:
            state = torch.zeros((batch_size, 1, self.d_model), device=x.device)
        x = x.unsqueeze(1) + state  # Add sequence dimension and combine with previous state
        x = self.transformer_encoder(x)
        state = x[:, -1, :].unsqueeze(1)  # Update state with the last hidden state
        
        x = self.fc(x).view(batch_size, self.chunk_size, -1)  # Reshape to (batch_size, chunk_size, num_classes)
        return x, state
