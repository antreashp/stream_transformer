import torch
import torch.nn as nn

class StreamingTransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, dropout, num_classes):
        super(StreamingTransformerModel, self).__init__()
        self.encoder = nn.Linear(input_dim, output_dim)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=output_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(output_dim, num_classes)
        self.memory = None

    def forward(self, x, reset_memory=False):
        # Reshape input x from (batch_size, channels, chunk_size) to (batch_size * channels, chunk_size)
        batch_size, channels, chunk_size = x.size()
        x = x.view(batch_size * channels, chunk_size)

        if reset_memory or self.memory is None:
            self.memory = self.encoder(x)
        else:
            new_memory = self.encoder(x)
            self.memory = torch.cat((self.memory, new_memory), dim=1)

        # Reshape memory back to (batch_size * channels, seq_len, output_dim)
        batch_size_channels, seq_len = self.memory.size()
        seq_len //= channels
        self.memory = self.memory.view(batch_size * channels, seq_len, -1)
        
        decoder_input = torch.zeros(batch_size * channels, seq_len, self.memory.size(-1), device=self.memory.device)
        output = self.transformer_decoder(decoder_input, self.memory)
        output = self.output_layer(output)
        
        # Reshape output back to (batch_size, channels, seq_len, num_classes)
        output = output.view(batch_size, channels, seq_len, -1)
        return output
