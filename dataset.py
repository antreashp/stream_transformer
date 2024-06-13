from helpers import *
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
class My_dataset(Dataset):
    def __init__(self, image_dir, annotation_dir, p):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.p = p
        self.image_ids = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.pt')]
        self.label_ids = [f.split('.')[0] for f in os.listdir(annotation_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label_id = self.label_ids[idx]
        image_path=f"{self.image_dir}/{image_id}.pt"
        annotation_file=f"{self.annotation_dir}/{label_id}.pt"
        inputs = torch.load(image_path)
        labels = torch.load(annotation_file)
        return torch.tensor(inputs), torch.tensor(labels), image_id#,( int(original_h[1:]), int(original_w[:-1]) )

class PascalVOCDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, p):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.p = p
        self.image_ids = [f.split('.')[0] for f in os.listdir(annotation_dir) if f.endswith('.png')]
        self.hilbert_curve, self.N = generate_hilbert_curve(self.p)
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = f"{self.image_dir}/{image_id}.jpg"
        annotation_file = f"{self.annotation_dir}/{image_id}.png"
        # Convert image and label to spectrograms
        image_curve, original_h, original_w = image_to_curve(image_path,self.hilbert_curve,self.N,label=False)
        label_curve, _, _,= image_to_curve(annotation_file,self.hilbert_curve,self.N,label=True)

        return torch.tensor(image_curve), torch.tensor(label_curve), image_id,( original_h, original_w )



if __name__ == '__main__':
    # Example usage
    image_dir = 'VOCdevkit/VOCdevkit/VOC2012/JPEGImages'
    segmentation_path = 'VOCdevkit/VOCdevkit/VOC2012/SegmentationClass'
    
    p = 9  # Adjust the order of the Hilbert curve as needed
    dataset = PascalVOCDataset(image_dir, segmentation_path, p)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # my_dataset = My_dataset('my_dataset/spectrograms','my_dataset/labels',p,nperseg,noverlap)
    # my_dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True)
    output_spectrogram_dir = 'my_dataset/input_curves'
    output_labels_dir = 'my_dataset/label_curves'
    os.makedirs(output_spectrogram_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    # for i, (spectrograms, labels, image_id) in enumerate(tqdm(dataloader,colour='green')):
    for i, (spectrograms, labels, image_id, image_shape) in enumerate(tqdm(dataloader)):
        # break
        # print(spectrograms.shape)
        # print(labels.shape)
        # print(image_shape)
        # exit()
        torch.save(spectrograms.squeeze(), f'{output_spectrogram_dir}/inputCurve_{int(image_shape[0]),int(image_shape[1])}_{image_id[0]}_{i}.pt')
        torch.save(labels.squeeze(), f'{output_labels_dir}/labelCurve_{int(image_shape[0]),int(image_shape[1])}_{image_id[0]}_{i}.pt')
                # Sanity check: Reconstruct the image from the spectrograms
        # print(f"Image ID: {image_id}")
        # if i>5:
        #     break
    exit()