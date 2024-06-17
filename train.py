import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import My_dataset
from model import StreamingTransformer
from helpers import *
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def chunk_input(input_tensor, chunk_size=512):
    batch_size, channels, time_steps = input_tensor.size()
    num_chunks = time_steps // chunk_size
    chunks = input_tensor.unfold(2, chunk_size, chunk_size)
    chunks = chunks.permute(0, 2, 1, 3).contiguous()
    return chunks.view(batch_size, num_chunks, channels, chunk_size)

def chunk_labels(label_tensor, chunk_size=512):
    batch_size, time_steps = label_tensor.size()
    num_chunks = time_steps // chunk_size
    chunks = label_tensor.unfold(1, chunk_size, chunk_size)
    return chunks.contiguous().view(batch_size, num_chunks, chunk_size)

def reassemble_chunks(chunks):
    batch_size, num_chunks, channels, chunk_size = chunks.size()
    chunks = chunks.permute(0, 2, 1, 3).contiguous()
    return chunks.view(batch_size, channels, num_chunks * chunk_size)

def train(model, dataloader, criterion, optimizer, device, chunk_size, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels, image_ids in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            original_h = int(image_ids[-1].split('_')[1].split(', ')[0][1:])
            original_w = int(image_ids[-1].split('_')[1].split(', ')[1][:-1])

            input_chunks = chunk_input(inputs, chunk_size)
            label_chunks = chunk_labels(labels, chunk_size)

            batch_size, num_chunks, channels, chunk_size = input_chunks.size()
            states = [torch.zeros((batch_size, 1, model.d_model), device=device) for _ in range(num_chunks)]
            all_outputs = []
            all_labels = []
            last_image_id = image_ids[-1]  # Keep track of the last image_id

            for i in range(num_chunks):
                chunk = input_chunks[:, i, :, :].to(device)
                label_chunk = label_chunks[:, i, :].to(device)

                optimizer.zero_grad()
                outputs, states[i] = model(chunk, states[i])
                outputs = softmax(outputs)
                all_outputs.append(outputs.detach().cpu())
                all_labels.append(label_chunk.detach().cpu())

                outputs = outputs.squeeze()
                label_chunk = label_chunk.squeeze().long()
                loss = criterion(outputs, label_chunk)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            all_outputs = torch.cat(all_outputs, dim=1)
            all_labels = torch.cat(all_labels, dim=1)
            last_output = all_outputs[-1].detach().cpu()
            last_label = all_labels[-1].detach().cpu()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

        last_output = torch.argmax(last_output, dim=1).unsqueeze(0)
        reconstructed_pred = curve_to_image(last_output, p, original_h, original_w)
        reconstructed_label = curve_to_image(last_label, p, original_h, original_w, label=True)

        plt.figure()
        plt.imshow(reconstructed_pred.squeeze(), cmap='tab20')
        plt.colorbar()
        plt.savefig(f'figures/reconstructed_pred.png')
        plt.figure()
        plt.imshow(reconstructed_label.squeeze(), cmap='tab20')
        plt.colorbar()
        plt.savefig(f'figures/reconstructed_label.png')

        foo = '_'.join(last_image_id.split("_")[-3:-1])
        image_path = f'{image_dir}/{foo}.jpg'
        labels_path = f'{annotation_dir}/{foo}.png'
        label_mask = parse_voc_annotation(labels_path)
        plt.figure()
        plt.imshow(label_mask.squeeze().detach().cpu().numpy(), cmap='tab20')
        plt.colorbar()
        plt.savefig(f'figures/label_mask.png')

        plt.figure()
        plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        plt.savefig(f'figures/original_image.png')

if __name__ == "__main__":
    input_dim = 512 * 512
    d_model = 256
    output_dim = 256
    num_heads = 32
    num_layers = 2
    dropout = 0.0
    chunk_size = 4096 * 32
    batch_size = 1
    learning_rate = 0.0005
    num_epochs = 100
    num_classes = 22
    p = 9

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = StreamingTransformer(chunk_size=chunk_size, num_classes=num_classes, d_model=d_model, nhead=num_heads, num_layers=num_layers).to(device)
    weight = torch.ones(num_classes).to(device)
    weight[0] = .01
    weight[21] = 1
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=0.0005)
    softmax = nn.Softmax(dim=2)

    image_dir = 'my_dataset_tiny/input_curves'
    annotation_dir = 'my_dataset_tiny/label_curves'
    dataset = My_dataset(image_dir, annotation_dir, p)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    image_dir = 'VOCdevkit/VOCdevkit/VOC2012/JPEGImages'
    annotation_dir = 'VOCdevkit/VOCdevkit/VOC2012/SegmentationClass'

    # for epoch in range(num_epochs):
    #     print(f'Epoch {epoch + 1}/{num_epochs}')
    train(model, dataloader, criterion, optimizer, device, chunk_size, num_epochs)
