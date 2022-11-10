import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import utils
from skimage import color
import cv2

color_map = np.random.randint(0, 255, 10 * 3)

def test_dataset(dataloader):
    for i_batch, sample_batched in enumerate(dataloader):
        print("testing dataset")
        # print("data value range:", torch.min(sample_batched[0]), torch.max(sample_batched[0]))

        if i_batch == 5:
            plt.figure()
            visualize_batch(sample_batched[0])
            plt.axis('off')
            plt.ioff()
            plt.savefig(str("sample_batch") + '.png')
            label = sample_batched[1].detach().numpy()[0]
            smap = visualize_seg_map(label)
            cv2.imwrite("seg_map_test.jpg", smap)
            break

def visualize_batch(tensor_batch):
    batch_size = len(tensor_batch)
    im_size = tensor_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(tensor_batch)
    grid = grid.numpy().transpose((1, 2, 0))
    blue = grid[:,:,0].copy()
    red = grid[:,:,2].copy()
    grid[:,:,2] = blue
    grid[:,:,0] = red
    plt.imshow(grid)

def visualize_seg_map(seg_map):

    canvas = np.ones((*seg_map.shape,3))

    for (x,y), value in np.ndenumerate(seg_map):
        canvas[x, y, 0] = color_map[int(value)]
        canvas[x, y, 1] = color_map[int(value * 2)]
        canvas[x, y, 2] = color_map[int(value * 3)]

    return canvas