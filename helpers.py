import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import warnings
warnings.filterwarnings('ignore')
import time
VOC_COLORMAP = np.array([
    [0, 0, 0],         # 0=background
    [128, 0, 0],       # 1=aeroplane
    [0, 128, 0],       # 2=bicycle
    [128, 128, 0],     # 3=bird
    [0, 0, 128],       # 4=boat
    [128, 0, 128],     # 5=bottle
    [0, 128, 128],     # 6=bus
    [128, 128, 128],   # 7=car
    [64, 0, 0],        # 8=cat
    [192, 0, 0],       # 9=chair
    [64, 128, 0],      # 10=cow
    [192, 128, 0],     # 11=dining table
    [64, 0, 128],      # 12=dog
    [192, 0, 128],     # 13=horse
    [64, 128, 128],    # 14=motorbike
    [192, 128, 128],   # 15=person
    [0, 64, 0],        # 16=potted plant
    [128, 64, 0],      # 17=sheep
    [0, 192, 0],       # 18=sofa
    [128, 192, 0],     # 19=train
    [0, 64, 128],      # 20=TV/monitor
    [224, 224, 192]    # 21=void
], dtype=np.uint8)

VOC_COLORMAP = np.array(VOC_COLORMAP, dtype=np.uint8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_hilbert_curve(order):
    N = 2 ** order
    hilbert_curve = HilbertCurve(order, 2)
    return hilbert_curve, N

def coordinates_to_hilbert_index(x, y, hilbert_curve):
    return hilbert_curve.distance_from_point([x, y])

def hilbert_index_to_coordinates(index, hilbert_curve):
    return hilbert_curve.point_from_distance(index)

def pad_image_to_square(image):
    c, h, w = image.shape
    max_dim = max(h, w)
    padded_image = torch.zeros((c, max_dim, max_dim), requires_grad=False, device=image.device)
    padded_image[:, :h, :w] = image
    return padded_image, h, w

def transform_image_to_1d(image, hilbert_curve, N):
    c, h, w = image.shape
    
    # Precompute the Hilbert indices for all coordinates
    coords = np.array([[i, j] for i in range(h) for j in range(w)])
    indices = [coordinates_to_hilbert_index(i, j, hilbert_curve) for i, j in coords]
    indices = torch.tensor(indices, device=image.device)
    
    # Create an empty tensor for the 1D image representation
    image_1d = torch.zeros((c, N * N), device=image.device)
    
    # Vectorized operation to gather pixel values based on precomputed indices
    for ch in range(c):
        image_1d[ch, indices] = image[ch, coords[:, 0], coords[:, 1]]
    
    return image_1d


def parse_voc_annotation(segmentation_path):
    segmentation_image = Image.open(segmentation_path).convert('RGB')
    label_mask = voc_label_indices(segmentation_image)
    label_mask = torch.from_numpy(label_mask).long().unsqueeze(0).to(device)
    return label_mask

def voc_label_indices(segmentation_image):
    segmentation_array = np.array(segmentation_image)
    label_mask = np.zeros(segmentation_array.shape[:2], dtype=np.uint8)
    for i, color in enumerate(VOC_COLORMAP):
        mask = np.all(segmentation_array == color, axis=-1)
        label_mask[mask] = i
    return label_mask
def load_image(image_path, label=False):
    if label:
        image = parse_voc_annotation(image_path)
    else:
        image = Image.open(image_path).convert('RGB')
        image = T.ToTensor()(image).to(device)
    return image
def map_1d_to_2d_image(image_1d, hilbert_curve, N, original_h, original_w):
    c, length = image_1d.shape
    padded_image = torch.zeros((c, N, N), device=image_1d.device)
    for ch in range(c):
        for idx in range(length):
            x, y = hilbert_index_to_coordinates(idx, hilbert_curve)
            if x < original_h and y < original_w:
                padded_image[ch, x, y] = image_1d[ch, idx]
    return padded_image[:, :original_h, :original_w]


def image_to_curve(image_path,hilbert_curve, N, label):
    # start = time.time()
    image_tensor = load_image(image_path, label=label)
    # print(f'Image loaded in {time.time() - start:.2f} seconds')
    # start = time.time()
    padded_image, original_h, original_w = pad_image_to_square(image_tensor)
    # print(f'Image padded in {time.time() - start:.2f} seconds')
    # start = time.time()
    image_1d = transform_image_to_1d(padded_image, hilbert_curve, N)
    # print(f'Image transformed to 1D in {time.time() - start:.2f} seconds')

    return image_1d, original_h, original_w

def curve_to_image(curve, order, original_h, original_w):
    hilbert_curve, N = generate_hilbert_curve(order)
    
    reconstructed_image_2d = map_1d_to_2d_image(curve, hilbert_curve, N, original_h, original_w)

    return reconstructed_image_2d

if __name__ == '__main__':
    image_path = 'VOCdevkit/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg'
    segmentation_path = 'VOCdevkit/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png'

    order = 9

    # Convert image to spectrogram (real and imaginary parts)
    image_curve, original_h, original_w = image_to_curve(image_path, order, label=False)
    reconstructed_image= curve_to_image(image_curve, order, original_h, original_w)

    # label_curve, original_h, original_w = image_to_curve(segmentation_path, order, label=True)
    # reconstructed_label= curve_to_image(label_curve, order, original_h, original_w)

    plt.figure()
    plt.imshow(reconstructed_image.permute(1, 2, 0).detach().cpu().numpy())
    # plt.figure()
    # plt.imshow(reconstructed_label.permute(1, 2, 0).detach().cpu().numpy())


    plt.show()