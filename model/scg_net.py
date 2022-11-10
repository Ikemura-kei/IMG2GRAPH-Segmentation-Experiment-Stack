import torch
import torch.nn as nn
from .scg_block import *
from .gcn_layer import *
import torch.nn.functional as F
from pretrainedmodels import se_resnext50_32x4d, se_resnext101_32x4d

import ssl
ssl._create_default_https_context = ssl._create_unverified_context # needed when downloading pretrained weights

class SCG_Net(nn.Module):

    def __init__(self, nb_nodes, nb_classes, gnn, dropout=0.2):
        super(SCG_Net, self).__init__()

        resnet = se_resnext50_32x4d()
        self.layer0, self.layer1, self.layer2, self.layer3, = \
            resnet.layer0, resnet.layer1, resnet.layer2, resnet.layer3

        self.conv0 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        for child in self.layer0.children():
            for param in child.parameters():
                par = param
                break
            break

        self.conv0.parameters = torch.cat([par[:, 0, :, :].unsqueeze(1), par], 1)
        self.layer0 = torch.nn.Sequential(self.conv0, *list(self.layer0)[1:4])

        self.scg_block = SCG_block(1024, nb_nodes, hidden_ch=nb_classes)

        self.node_size = nb_nodes

        self.training = True
        
        self.gnn = gnn

        # self.graph_layers1 = GCN_Layer(1024, out_features=128, bnorm=True, activation=nn.ReLU(True), dropout=dropout)

        # self.graph_layers2 = GCN_Layer(128, out_features=nb_classes, bnorm=False, activation=None)

        self.num_out_channels = nb_classes

        self.node_size = nb_nodes

    def forward(self, x):
        x_size = x.size()

        # gx = self.pretrained_resnet50(x) # of shape (B, 256, 32, 32)
        gx = self.layer3(self.layer2(self.layer1(self.layer0(x))))

        A, gx, loss, z_hat = self.scg_block(gx)
        # gx has shape (B, 256, node_size[0], node_size[1])
        # A has shape (B, node_size[0]*node_size[1], node_size[0]*node_size[1])
        B, C, H, W = gx.size()

        x_size = x.size()

        # print("x size:", x.size())
        # print("gx size:", gx.size())
        # print("A size:", A.size())
        # print("gx view:",gx.view(B, -1, C).size())
        # print("gx size after gnn:", gx.size())

        # gx, _= self.graph_layers2(self.graph_layers1((gx.view((B, -1, C)), A)))
        gx, _ = self.gnn((gx.view((B, -1, C)), A))

        gx = gx + z_hat

        gx = gx.view(gx.size()[0], self.num_out_channels, self.node_size[0], self.node_size[1])

        # gx = F.interpolate(gx, (H, W), mode='bilinear', align_corners=False)

        if self.training:
            return F.interpolate(gx, x_size[2:], mode='bilinear', align_corners=False), loss
        else:
            return F.interpolate(gx, x_size[2:], mode='bilinear', align_corners=False)