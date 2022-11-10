import torch
from torch.utils.data import Dataset
import cv2 as cv
from os.path import join
import os
from PIL import Image
import numpy as np

class NYUDepthV2Dataset(Dataset):

    def __init__(self, require_depth, data_dir, image_size, transform_func=None):
        self.require_depth = require_depth
        self.root_dir = data_dir
        self.transform_func = transform_func
        self.image_size = image_size

        if self.require_depth:
            self.depth_paths = []
        else:
            self.depth_paths = None
        
        self.rgb_paths = []
        self.seg_map_paths = []

        for idx, file in enumerate(os.listdir(join(self.root_dir, "rgb"))):
            self.rgb_paths.append(join("rgb", file))
            self.seg_map_paths.append(join("seg_map", os.path.splitext(file)[0] + '.npy'))
            if self.depth_paths is not None:
                self.depth_paths.append(join("depth", file))

        print("%d images detected" % (len(self.rgb_paths)))
        if self.depth_paths is not None:
            print("%d depths detected" % (len(self.depth_paths)))

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        rgb_image = cv.imread(join(self.root_dir, self.rgb_paths[index]))
        rgb_image = Image.fromarray(rgb_image)

        if self.require_depth:
            depth_map = cv.imread(join(self.root_dir,self.depth_paths[index]), cv2.IMREAD_GRAYSCALE)
            depth_map = Image.fromarray(depth_map)

        if self.transform_func:
            rgb_image = self.transform_func(rgb_image)
            if self.require_depth:
                depth_map = self.transform_func(depth_map)

        segmentation_map = np.load(join(self.root_dir, self.seg_map_paths[index]))
        segmentation_map = cv.resize(segmentation_map, self.image_size, interpolation=cv.INTER_NEAREST)
        segmentation_map = np.expand_dims(segmentation_map, axis=2)
        segmentation_map = torch.from_numpy(segmentation_map).permute(2, 0, 1)
        segmentation_map = torch.squeeze(segmentation_map, 0)
        
        if self.require_depth:
            return rgb_image, depth_map, segmentation_map
        else:
            return rgb_image, segmentation_map