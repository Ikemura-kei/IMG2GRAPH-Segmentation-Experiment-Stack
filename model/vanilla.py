import torch
import torch.nn as nn
import numpy as np

class Vanilla(nn.Module):
    def __init__(self, num_classes, gnn, nodes):
        super(Vanilla, self).__init__()

        self.num_nodes = nodes[0] * nodes[1]
        self.nodes = nodes

    def forward(self, data):
        B, C, H, W = data.size()

        A = torch.zeros((B, int(self.num_nodes), int(self.num_nodes)))
        self.scale = int(W / nodes[0])

        for b in range(B):
            for x in range(int(W / self.scale)):
                for y in range(int(H / self.scale)):

                    selfx1, selfx2, selfy1, selfy2 = self.scaled_coord_to_img_coord(x, y, self.scale)
                    self_node_id = self.scaled_coord_to_node_number(x, y, W)

                    if y > 0:
                        up = (x, y-1)
                        upx1, upx2, upy1, upy2 = self.scaled_coord_to_img_coord(up[0], up[1], self.scale)
                        up_node_id = self.scaled_coord_to_node_number(up[0], up[1], W)
                        A[b, self_node_id, up_node_id] = self.affinity(data[b, :, selfy1:selfy2, selfx1:selfx2], data[b, :, upy1:upy2, upx1:upx2])

                    if y < H - 1:
                        down = (x, y+1)
                        downx1, downx2, downy1, downy2 = self.scaled_coord_to_img_coord(down[0], down[1], self.scale)
                        down_node_id = self.scaled_coord_to_node_number(down[0], down[1], W)
                        print(data[b, :, selfy1:selfy2, selfx1:selfx2].size())
                        print(data[b, :, downy1:downy2, downx1:downx2].size())
                        A[b, self_node_id, down_node_id] = self.affinity(data[b, :, selfy1:selfy2, selfx1:selfx2], data[b, :, downy1:downy2, downx1:downx2])

                    if x < 0:
                        left = (x-1, y)
                        leftx1, leftx2, lefty1, lefty2 = self.scaled_coord_to_img_coord(left[0], left[1], self.scale)
                        left_node_id = self.scaled_coord_to_node_number(left[0], left[1], W)
                        A[b, self_node_id, left_node_id] = self.affinity(data[b, :, selfy1:selfy2, selfx1:selfx2], data[b, :, lefty1:lefty2, leftx1:leftx2])

                    if x < W - 1:
                        right = (x+1, y)
                        rightx1, rightx2, righty1, righty2 = self.scaled_coord_to_img_coord(right[0], right[1], self.scale)
                        right_node_id = self.scaled_coord_to_node_number(right[0], right[1], W)
                        A[b, self_node_id, right_node_id] = self.affinity(data[b, :, selfy1:selfy2, selfx1:selfx2], data[b, :, righty1:righty2, rightx1:rightx2])
      
        if self.training:
            return A, data, None
        else:
            return A, data

    @classmethod
    def scaled_coord_to_img_coord(self, x, y, scale):
        return x * scale, x * scale + scale - 1, y * scale, y * scale + scale - 1
    
    @classmethod
    def scaled_coord_to_node_number(self, x, y, row_number):
        return x + y * row_number

    @classmethod
    def affinity(self, src, dst):
        return torch.sum(torch.absolute(src - dst))
