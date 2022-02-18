import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self,num_classes):
        super(ResNet, self).__init__()
        self.resnet=models.resnet101(pretrained=True)
        self.fc=nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, num_classes)
        )

    def forward(self,input):
        x=self.resnet(input)
        x=self.fc(x)
        return x

class VGG(nn.Module):
    def __init__(self,num_classes):
        super(VGG, self).__init__()
        self.vgg=models.vgg16(pretrained=True)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, num_classes)
        )

    def forward(self,input):
        x=self.vgg(input)
        x=self.fc(x)
        return x