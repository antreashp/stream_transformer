import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Assuming model.py contains the Streaming Transformer model definition
from model import StreamingTransformerModel

def train(model, dataloader, criterion, optimizer, device, chunk_size):
    model.train()
    for i, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Reshape inputs and targets to the correct dimensions
        inputs = inputs.view(inputs.size(0), 3, 512 * 512)
        targets = targets.view(targets.size(0), 512 * 512)
        
        # Reset model memory at the start of each new sequence
        model.memory = None
        
        # Process inputs in chunks
        for j in range(0, inputs.size(2), chunk_size):
            chunk_input = inputs[:, :, j:j+chunk_size].to(device)
            chunk_target = targets[:, j:j+chunk_size].to(device)
            
            # Ensure the chunk size matches the model's expected input
            if chunk_input.size(2) < chunk_size:
                padding_size = chunk_size - chunk_input.size(2)
                chunk_input = torch.cat([chunk_input, torch.zeros(chunk_input.size(0), chunk_input.size(1), padding_size).to(device)], dim=2)
                chunk_target = torch.cat([chunk_target, torch.zeros(chunk_target.size(0), padding_size).to(device)], dim=1)

            output = model(chunk_input, reset_memory=(j == 0))  # Forward pass
            output = output.view(-1, output.size(-1))  # Flatten for classification
            chunk_target = chunk_target.view(-1)  # Flatten targets

            loss = criterion(output, chunk_target)  # Compute loss
            
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            print(f'Processed chunk {j//chunk_size + 1}/{inputs.size(2)//chunk_size}: loss {loss.item()}')

if __name__ == "__main__":
    # Hyperparameters
    input_dim = 512 * 512
    output_dim = 512
    num_heads = 8
    num_layers = 6
    dropout = 0.1
    chunk_size = 64
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    num_classes = 21
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model, criterion, optimizer
    model = StreamingTransformerModel(input_dim, output_dim, num_heads, num_layers, dropout, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Dummy dataset and dataloader
    dataset = torch.utils.data.TensorDataset(torch.randn(1024, 3, 512 * 512), torch.randint(0, num_classes, (1024, 512 * 512)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train the model
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train(model, dataloader, criterion, optimizer, device, chunk_size)
