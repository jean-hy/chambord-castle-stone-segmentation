import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class StoneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory with image patches.
            mask_dir (str): Path to the directory with corresponding masks.
            transform (callable, optional): A function/transform to apply to both image and mask.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))  

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

class JointTransform:
    """
    Example joint transform for images and masks.
    We can add more augmentations here (e.g. random flips, color jitter).
    """
    def __init__(self, resize=(256, 256)):
        self.image_transform = T.Compose([
            T.Resize(resize),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = T.Compose([
            T.Resize(resize, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()  # Will produce a 0-1 mask
        ])

    def __call__(self, image, mask):
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        return image, mask
