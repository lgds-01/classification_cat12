import torch.nn as nn
import torch
import einops

class MLP(nn.Module):
    def __init__(self,num_class):
        super(MLP, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(224*224*3,1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128,num_class),
            nn.Softmax(dim=1)
        )
    def forward(self,input):
        batch_size=input.size(0)
        x=input.view(batch_size,-1)
        return self.net(x)

class CNN(nn.Module):
    def __init__(self,num_class):
        super(CNN, self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,8,(3,3),1,1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(3,2),
            nn.Conv2d(8,16,(3,3),1,1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(3,2),
            nn.Conv2d(16,32,(3,3),1,1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3,2),
            nn.Conv2d(32,64,(3,3),1,1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3,2),
            nn.Conv2d(64,128,(3,3),1,1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3,2),
            nn.Flatten(),
            nn.Linear(128*6*6,12),
            nn.Softmax(dim=-1),
        )

    def forward(self,input):
        x=self.net(input)
        return x

class VGG(nn.Module):
    def __init__(self,cfg,num_classes=1000):
        super(VGG, self).__init__()
        self.num_classes=num_classes

        self.features=self._make_layers(cfg)
        self.avgpool=nn.AdaptiveAvgPool2d((7,7))
        self.classifier=nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,num_classes),
        )

    def _make_layers(self,cfg):
        layers=[]
        in_channels=3
        for v in cfg:
            if v=='M':
                layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                conv2d=nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
                layers+=[conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
                in_channels=v
        return nn.Sequential(*layers)

    def forward(self,input):
        batch_size=input.shape[0]
        x=self.features(input)
        x=self.avgpool(x)
        x=x.view(batch_size,-1)
        x=self.classifier(x)
        return x

class BottleNeck(nn.Module):
    # bottleneck层输出通道是输入的4倍
    expansion=4
    def __init__(self,inplanes,planes,stride=1,downsaple=None,groups=1,base_width=64,dilation=1):
        super(BottleNeck, self).__init__()
        width=int(planes*(base_width/64.))*groups

        self.conv1=nn.Conv2d(inplanes,width,(1,1),bias=False)
        self.bn1=nn.BatchNorm2d(width)
        self.conv2=nn.Conv2d(width,width,(3,3),stride,padding=dilation,groups=groups,bias=False,dilation=dilation)
        self.bn2=nn.BatchNorm2d(width)
        self.conv3=nn.Conv2d(width,planes*self.expansion,(1,1),bias=False)
        self.bn3=nn.BatchNorm2d(planes*self.expansion)
        self.relu=nn.ReLU(inplace=True)
        self.downsaple=downsaple
        self.stride=stride

    def forward(self,input):
        identity=input

        x=self.relu(self.bn1(self.conv1(input)))
        x=self.relu(self.bn2(self.conv2(x)))
        x=self.relu(self.bn3(self.conv3(x)))

        if self.downsaple is not None:
            identity=self.downsaple(input)

        x=x+identity
        x=self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self,num_classes,layer):
        super(ResNet, self).__init__()
        self.inplanes=64

        self.conv1=nn.Conv2d(3,self.inplanes,(7,7),stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.inplanes)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(3,2,1)

        self.layer1=self._make_layers(64,layer[0])
        self.layer2=self._make_layers(128,layer[1],stride=2)
        self.layer3=self._make_layers(256,layer[2],stride=2)
        self.layer4=self._make_layers(512,layer[3],stride=2)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Sequential(
            nn.Linear(512*4,1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000,num_classes),
        )

    def _make_layers(self,planes,blocks,stride=1):
        if stride!=1 or self.inplanes!=planes*4:
            downsaple=nn.Sequential(
                nn.Conv2d(self.inplanes,planes*4,(1,1),stride,bias=False),
                nn.BatchNorm2d(planes*4),
            )

        layers=[]
        layers.append(BottleNeck(self.inplanes,planes,stride,downsaple))
        self.inplanes=planes*4
        for _ in range(1,blocks):
            layers.append(BottleNeck(self.inplanes,planes))

        return nn.Sequential(*layers)

    def forward(self,input):
        x=self.maxpool(self.relu(self.bn1(self.conv1(input))))

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.fc(x)

        return x

class ViT(nn.Module):
    def __init__(self,image_size,patch_size,num_classes,depth,heads,mlp_dim):
        super(ViT, self).__init__()
        num_patches=(image_size//patch_size)**2
        hidden_size=3*patch_size**2
        self.patch_size=patch_size
        self.hidden_size=hidden_size
        self.embdding=nn.Conv2d(3,hidden_size,patch_size,patch_size)
        self.pos_embdding=nn.Parameter(torch.randn(1,num_patches+1,hidden_size))
        self.cls=nn.Parameter(torch.randn(1,1,hidden_size))
        self.transformer=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size,heads,mlp_dim),
            depth
        )
        self.to_cls_token=nn.Identity()
        self.mlp_head=nn.Linear(hidden_size,num_classes)

    def forward(self,input):
        x=self.embdding(input)
        x=einops.rearrange(x,'b c h w -> b (h w) c')
        b,n,_=x.shape
        cls_tokens=einops.repeat(self.cls,'() n d -> b n d',b=b)
        x=torch.cat((cls_tokens,x),dim=1)
        x+=self.pos_embdding[:,:(n+1)]
        x=self.transformer(x)
        x=self.to_cls_token(x[:,0])
        x=self.mlp_head(x)
        return x


