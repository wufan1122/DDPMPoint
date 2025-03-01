import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import data.util as Util

class hpatchesDataset(Dataset):
    def __init__(self, root_dir, folder_type='i', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and homography files.
            folder_type (string): 'i' or 'v' to choose which folder type to load images from.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.folder_type = folder_type
        self.transform = transform
        self.samples = []

        # List all subdirectories
        for subdir in os.listdir(os.path.join(root_dir, folder_type)):
            subdir_path = os.path.join(root_dir, folder_type, subdir)
            images = []
            homographies = []
            if os.path.isdir(subdir_path):
                # List all files in the subdirectory
                files = os.listdir(subdir_path)
                img_files = [f for f in files if f.endswith('.png')]
                H_files = [f for f in files if f.startswith('H_')]

                # Sort files to ensure consistent order
                img_files.sort()
                H_files.sort()

                # Load images and homographies
                for f in img_files:
                    img = Image.open(os.path.join(subdir_path, f)).convert("RGB")
                    img = img.resize((640, 480), Image.Resampling.LANCZOS)
                    img = Util.transform_augment_cd(img, 'val', min_max=(-1, 1))
                    images.append(img)
                for f in H_files:
                    h = np.loadtxt(os.path.join(subdir_path, f)).reshape(3, 3)
                    h = torch.tensor(h, dtype=torch.float32)  # Convert to tensor
                    homographies.append(h)
                # Apply transform if specified

            # Add the sample to the list of samples
            self.samples.append({
                'img1': images[0],
                'img2': images[1],
                'img3': images[2],
                'img4': images[3],
                'img5': images[4],
                'img6': images[5],
                'H12': homographies[0],
                'H13': homographies[1],
                'H14': homographies[2],
                'H15': homographies[3],
                'H16': homographies[4],
                'subdir': subdir
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
