from torchmetrics.classification import Accuracy
import torch
import pandas as pd
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

batch_size=8
cpu=torch.device('cpu')
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform=transforms.Compose([
    transforms.ToTensor(),
])


class TrainDataset(Dataset):
    def __init__(self,data_path,transform=None):
        self.transform=transform
        with open(data_path,'r') as f:
            items=f.readlines()
            self.img_paths=[]
            self.labels=[]
            for item in items:
                path,label=str.split(item)
                self.img_paths.append(path)
                self.labels.append(label)

    def __getitem__(self, index):
        img_path=os.path.join('cat_12',self.img_paths[index])
        img=Image.open(img_path,'r').resize((224,224))
        img=np.array(img).astype('float32')
        if self.transform is not None:
            img=self.transform(img)
        return img,np.array(self.labels[index]).astype('int64')

    def __len__(self):
        return len(self.labels)

def train(model,dataloader,loss_func,optim,epochs,model_path,pre_model_path=None):
    if pre_model_path is not None:
        model=torch.load(pre_model_path)
    accuracy=Accuracy()
    model.train()
    for epoch in range(epochs):
        for id,(data,label) in enumerate(dataloader):
            data,label=data.to(device),label.to(device)
            pred=model(data)
            loss=loss_func(pred,label)
            optim.zero_grad()
            loss.backward()
            optim.step()

            pred,label=pred.to(cpu),label.to(cpu)
            accuracy.update(pred,label)
            if id % 50 == 0:
                acc=accuracy.compute()
                print('epoch {}\tid {}\tloss {}\tacc {}'.format(epoch,id,loss.item(),acc))

            torch.cuda.empty_cache()

        print('{}'.format('-'*40))
        total_acc=accuracy.compute()
        accuracy.reset()
        print('epoch {} done! total_acc {}'.format(epoch,total_acc))
        print('{}'.format('-'*40))

        torch.save(model,model_path)

def predict(model_path,test_path,result_path):
    model=torch.load(model_path)
    model.eval()
    preds=[]
    for img_path in os.listdir(test_path):
        img=Image.open(os.path.join(test_path,img_path),'r').resize((224,224))
        data=np.array(img).astype('float32')
        data=transform(data).unsqueeze(0).to(device)
        pred=model(data)
        pred=torch.argmax(pred,dim=1)
        preds.append(pred.item())
    preds=np.reshape(preds,(-1))
    data=zip(os.listdir(test_path),preds)
    df=pd.DataFrame(data,columns=['Path','Label'])
    df.to_csv(result_path,header=False,index=False)
    print("generate test dataset label done!")
