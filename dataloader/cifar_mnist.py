from torch.utils.data import Dataset
import os
import cv2 as cv
from PIL import Image
import numpy as np
import torch

class CifarMnist(Dataset):
    def __init__(self, data_root, target_img_shape, transform=None):
        super(CifarMnist, self).__init__()

        self.data_root = data_root
        self.transform = transform
        self.image_size = target_img_shape

        self.rgb_dir = os.path.join(self.data_root, "rgb")
        self.seg_map_dir = os.path.join(self.data_root, "seg_map")

        self.rgb_paths = []
        self.seg_map_paths = []

        for idx, f in enumerate(os.listdir(self.rgb_dir)):
            self.rgb_paths.append(os.path.join(self.rgb_dir, f))
            self.seg_map_paths.append(os.path.join(self.seg_map_dir, os.path.splitext(f)[0] + ".npy"))


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = torch.tolist(index)

        rgb = cv.imread(self.rgb_paths[index])
        rgb = Image.fromarray(rgb) # convert to PIL image for transforms to apply

        if self.transform is not None:
            rgb = self.transform(rgb)

        label = np.load(self.seg_map_paths[index])
        label = cv.resize(label, self.image_size, interpolation=cv.INTER_NEAREST)
        label = torch.from_numpy(label)

        return rgb, label

    def __len__(self):
        return len(self.rgb_paths)
