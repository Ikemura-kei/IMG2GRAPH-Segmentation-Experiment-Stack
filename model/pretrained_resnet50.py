from torchvision import models
import torch.nn as nn

class PretrainedWideResnet50(nn.Module):

    def __init__(self):
        super(PretrainedWideResnet50, self).__init__()

        wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)

        for param in wide_resnet50_2.parameters():
            param.requires_grad = False
            
        self.layer1 = nn.Sequential(*list(wide_resnet50_2.children())[0:4])
        self.layer2_path1 = nn.Sequential(*list((list((list(wide_resnet50_2.children())[4]).children()))[0].children())[0:7])
        self.layer2_path2 = nn.Sequential(list((list((list(wide_resnet50_2.children())[4]).children()))[0].children())[7])

    def forward(self, x):
        x1 = self.layer1(x)
        return self.layer2_path1(x1) + self.layer2_path2(x1)
