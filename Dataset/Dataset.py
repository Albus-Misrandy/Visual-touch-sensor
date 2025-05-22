import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.Visualization import *


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = numpy_to_tensor(image)
        mask = numpy_to_tensor(mask)

        # 二值化mask
        mask = (mask > 0.1).float()

        return image, mask


class UnpressImageDataset(Dataset):
    def __init__(self, image_dir):
        self.img_dir = image_dir
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        image = Image.open(img_path).convert("RGB")
        image = numpy_to_tensor(image)

        return image