import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from utils import train,TrainDataset
import models
import pre_models

batch_size=16
transform=transforms.Compose([
    transforms.ToTensor(),
])
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_dataset=TrainDataset('cat_12/train_list.txt',transform)
train_dataloader=DataLoader(train_dataset,batch_size,shuffle=True)

# MLP
# mlp=models.MLP(num_class=12).to(device)
# optimzer=torch.optim.Adam(params=mlp.parameters(),lr=0.0001)
# loss_func=nn.CrossEntropyLoss()
# model_path='weights/mlp.pth'
# train(mlp,train_dataloader,loss_func,optimzer,70,model_path)

# CNN
# cnn=models.CNN(num_class=12).to(device)
# optimzer=torch.optim.Adam(params=cnn.parameters(),lr=0.0001)
# loss_func=nn.CrossEntropyLoss()
# model_path='weights/cnn.pth'
# train(cnn,train_dataloader,loss_func,optimzer,60,model_path)

# VGG
# cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
# vgg=models.VGG(cfg,num_classes=12).to(device)
# optimzer=torch.optim.Adam(params=vgg.parameters(),lr=0.0001)
# loss_func=nn.CrossEntropyLoss()
# model_path='weights/vgg.pth'
# train(vgg,train_dataloader,loss_func,optimzer,10,model_path,pre_model_path=model_path)

# pretrained VGG
# vgg=pre_models.VGG(num_classes=12).to(device)
# optimzer=torch.optim.Adam(params=vgg.parameters(),lr=0.0001)
# loss_func=nn.CrossEntropyLoss()
# model_path='weights/vgg_pre.pth'
# train(vgg,train_dataloader,loss_func,optimzer,40,model_path)

# ResNet
# resnet=models.ResNet(num_classes=12,layer=[3,4,6,3]).to(device)
# optimzer=torch.optim.Adam(params=resnet.parameters(),lr=0.0001)
# loss_func=nn.CrossEntropyLoss()
# model_path='weights/resnet.pth'
# train(resnet,train_dataloader,loss_func,optimzer,40,model_path)

# pretrianed ResNet
resnet=pre_models.ResNet(num_classes=12).to(device)
optimzer=torch.optim.Adam(params=resnet.parameters(),lr=0.0001)
loss_func=nn.CrossEntropyLoss()
model_path='weights/resnet_pre.pth'
train(resnet,train_dataloader,loss_func,optimzer,40,model_path)

# ViT
# vit=models.ViT(224,32,12,4,4,512).to(device)
# optimzer=torch.optim.Adam(params=vit.parameters(),lr=0.0001)
# loss_func=nn.CrossEntropyLoss()
# model_path='weights/vit.pth'
# train(vit,train_dataloader,loss_func,optimzer,60,model_path)