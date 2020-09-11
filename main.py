from __future__ import absolute_import

from argparse import ArgumentParser
import gc
import os
import numpy as np
import optuna
from PIL import Image
from pytorch_metric_learning import losses, samplers
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from random import sample, random
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, Dataset


train_path = '/lab/vislab/OPEN/datasets_RGB_one/train/'
valid_path = '/lab/vislab/OPEN/datasets_RGB_one/val/'
num_classes = 100


class Model(LightningModule):
    """ Model 
    """
    #TODO: it might be wise to implement specific __getitem__ functions or something similar to that

    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.epoch = 0
        self.learning_rate = 0.001
        self.loss = nn.CrossEntropyLoss()
        self.training_correct_counter = 0
        self.training = False

        mod1=models.resnet50(pretrained=True)
        mod2=models.resnet50(pretrained=True)
        self.model1 = nn.Sequential(
            mod1.conv1,
            mod1.bn1,
            mod1.relu,
            mod1.maxpool,

            mod1.layer1,
            mod1.layer2,
            mod1.layer3,
            mod1.layer4,
        )
        self.model2 = nn.Sequential(
            mod2.conv1,
            mod2.bn1,
            mod2.relu,
            mod2.maxpool,

            mod2.layer1,
            mod2.layer2,
            mod2.layer3,
            mod2.layer4,
        )
        self.fc=nn.Linear(4096,num_classes)


    def forward(self, x):
        x=x.transpose(1,0)
        x0=x[:-1].transpose(1,0)
        x1=x[-1]
        bs, ncrops, c, h, w = x0.size()
        x0=x0.contiguous().view((-1, c, h, w))
        x0 = self.model1(x0)
        x0 = F.avg_pool2d(x0, 8)
        x0,_ = torch.max(x0.view(bs, ncrops, -1),1)
        x1= self.model2(x1)
        x1 = F.avg_pool2d(x1, 8)
        x1=x1.view(bs,-1)
        x=torch.cat([x0,x1],1)
        if self.training==True:
            x=F.dropout(x,0.4)
        return self.fc(x.view(x.size(0), -1))

    def prepare_data(self):

        data_transforms = {
            'train': transforms.Compose([

            transforms.Lambda(lambda img: self.RandomErase(img)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Lambda(lambda img: self.crops_and_random(img)),
            #transforms.Resize((512,512),interpolation=2),
            #transforms.Lambda(lambda img: four_and_random(img)),
            transforms.Lambda(lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Normalize(mean=[n / 255.
                                                                                                        for n in
                                                                                                        [129.3, 124.1,
                                                                                                        112.4]],
                                                                                                std=[n / 255. for n in
                                                                                                    [68.2, 65.4,
                                                                                                        70.4]])])(crop) for
                                                        crop in crops]))

            ]),


            'valid': transforms.Compose([
            #transforms.Resize((512,512),interpolation=2),
            #transforms.FiveCrop(256),
            transforms.Lambda(lambda img: self.val_crops(img)),
            transforms.Lambda(lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Normalize(mean=[n / 255.
                                                                                                        for n in
                                                                                                        [129.3, 124.1,
                                                                                                        112.4]],
                                                                                                std=[n / 255. for n in
                                                                                                    [68.2, 65.4,
                                                                                                        70.4]])])(crop) for
                                                        crop in crops]))

            ]),
        }   

        self.trainset = datasets.ImageFolder(train_path, data_transforms['train'])
        self.validset = datasets.ImageFolder(valid_path, data_transforms['valid'])
        print("self.validset.length", self.validset)


    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=32, sampler=None, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=32, sampler=None, num_workers=4)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        self.training = True
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss(outputs, labels)

        return {'loss': loss}
    
    def training_epoch_end(self, training_step_outputs):
        self.training = True
        train_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        return {
            'log': {'train_loss': train_loss},
            'progress_bar': {'train_loss': train_loss}
        }

    def validation_step(self, batch, batch_idx):
        self.training = False
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss(outputs, labels)

        labels_hat = torch.argmax(outputs, dim=1)
        val_acc = torch.sum(labels == labels_hat).item() / (len(labels) * 1.0)


        return {
            'val_loss': loss,
            'val_acc': val_acc
        }
    
    def validation_epoch_end(self, validation_step_outputs):
        self.training = False
        val_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        val_acc = np.mean([x['val_acc'] for x in validation_step_outputs])
        print("val_loss", val_loss)
        print("val_acc", val_acc)

        self.epoch += 1
        return {
            'log': {
                'val_loss': val_loss,
                'val_acc': val_acc
                },
            'progress_bar': {
                'val_loss': val_loss,
                'val_acc': val_acc
            }
        }
    
    def random_crops(self, img, k):
        crops=[]
        five256=torchvision.transforms.RandomCrop(256)
        for j in range(k):
            im = five256(img)
            crops.append(im)
        return crops

    def crops_and_random(self, img):
        im=np.array(img)
        w, h, _ = im.shape
        if w<h:
            wi=512
            hi=int(wi*h*1.0/w)
        else:
            hi=512
            wi = int(hi * w * 1.0 / h)

        res=torchvision.transforms.Resize((wi,hi),interpolation=2)
        img=res(img)
        Rand=torchvision.transforms.RandomCrop(512)
        re=torchvision.transforms.Resize((256,256),interpolation=2)
        return self.random_crops(img, 4)+[re(Rand(img))]

    def val_crops(self, img):
        im = np.array(img)
        w, h, _ = im.shape
        if w < h:
            wi = 512
            hi = int(wi * h * 1.0 / w)
        else:
            hi = 512
            wi = int(hi * w * 1.0 / h)
        res=torchvision.transforms.Resize((wi,hi),interpolation=2)
        img=res(img)
        im=np.array(img)
        Rand = torchvision.transforms.RandomCrop(512)
        re = torchvision.transforms.Resize((256, 256), interpolation=2)
        a=int(wi/256)
        b=int(hi/256)
        crs=[]
        for i in range(a):
            for j in range(b):
                crs.append(Image.fromarray((im[i*256:((i+1)*256),j*256:((j+1)*256)]).astype('uint8')).convert('RGB'))
        return sample(crs,4)+[re(Rand(img))]

    def RandomErase(self, img, p=0.5, s=(0.06,0.12), r=(0.5,1.5)):
        im=np.array(img)
        w,h,_=im.shape
        S=w*h
        pi=random()
        if pi>p:
            return img
        else:
            Se=S*(random()*(s[1]-s[0])+s[0])
            re=random()*(r[1]-r[0])+r[0]
            He=int(np.sqrt(Se*re))
            We=int(np.sqrt(Se/re))
            if He>=h:
                He=h-1
            if We>=w:
                We=w-1
            xe=int(random()*(w-We))
            ye=int(random()*(h-He))
            im[xe:xe+We,ye:ye+He]=int(random()*255)
            return Image.fromarray(im.astype('uint8')).convert('RGB')

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.1)
        return parser


if __name__ == '__main__':
    trainer = pl.Trainer(
        max_epochs=25,
        num_sanity_val_steps=-1,
        gpus=[2] if torch.cuda.is_available() else None
    ) 

    model_ft = Model()
    ct=0
    for child in model_ft.model1.children():
        ct += 1
        if ct < 6: # change to lower numbers for training more layers
            for param in child.parameters():
                param.requires_grad = False
    ct=0
    for child in model_ft.model2.children():
        ct += 1
        if ct < 6:
            for param in child.parameters():
                param.requires_grad = False


    trainer.fit(model_ft)