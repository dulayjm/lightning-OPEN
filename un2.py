from __future__ import absolute_import

import neptune

# neptune.init('dulayjm/sandbox')
# neptune.create_experiment(name='rgb-10classes-pyramid', params={'lr': 0.01}, tags=['resnet', 'iNat', 'pyramid'])

from argparse import ArgumentParser
import gc
import os
import math
import numpy as np
import optuna
# import matplotlib.pyplot as plt
from PIL import Image
from pytorch_metric_learning import losses, samplers
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Callback
from pytorch_lightning import loggers as pl_loggers
from random import sample, random
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, Dataset

############################################################
# Script to run this on the OPEN datasets


train_path = '/lab/vislab/OPEN/datasets_RGB_new/train/' # new is the 10 class one
valid_path = '/lab/vislab/OPEN/datasets_RGB_new/val/'

num_classes = 10


def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] // out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] // out_pool_size[i]))
        h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)//2 # maybe try to implement outlier pooling on this stuff  
        w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)//2



        print("STUFF")
        print(h_wid, w_wid, h_pad, w_pad)
        # 31 31 0 0

        # 1/0

        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        print('maxpool shape', x.shape)

        if(i == 0):
            spp = x.view(num_sample,-1) 
            print("spp size:",spp.size())
        else:
            print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp


class Model(LightningModule):
    """ Model 
    """

# Use a pre-trained ResNet!!!

    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.epoch = 0
        self.learning_rate = 0.01
        self.loss = nn.CrossEntropyLoss()
        self.training = False
        nc = 3
        ndf = 64
        self.output_num = [4,2,1]

        mod = models.resnet50(pretrained=True)
        self.mod = nn.Sequential(
            mod.conv1,
            mod.bn1,
            mod.relu,
            mod.maxpool,

            mod.layer1,
            mod.layer2,
            mod.layer3,
            mod.layer4,
        )
        self.fc1 = nn.Linear(43008, 10000)
        self.fc2 = nn.Linear(10000, 4096)
        self.fc3 = nn.Linear(4096,10) # may switch to 10 ie num_classes 
                                    # or add anohtehr fc layer ...

    def forward(self,x):
        x=x.transpose(1,0)
        x=x[-1]
        print("x.shape before the model call", x.shape)

        x=self.mod(x)
        print("x.shape before the spp call", x.shape)
        spp = spatial_pyramid_pool(x,15,[int(x.size(2)),int(x.size(3))],self.output_num)
        # fc layer?
        print("spp.shape before pass to fc layer", spp.shape)
        print(spp.size)
        fc1 = self.fc1(spp)
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)
        s = nn.Sigmoid()
        output = s(fc3)
        return output



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


# currently same as train 
            'valid': transforms.Compose([
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


    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=15, shuffle=True, drop_last=True, sampler=None, num_workers=15)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=15, shuffle=True, drop_last=True, sampler=None, num_workers=15)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        # self.model1.train()
        # self.model2.train()

        self.train()
        
        # for imgs, labels in model_ft.trainset: 
        #     print(labels)

        self.training = True
        inputs, labels = batch

        outputs = self(inputs)
        loss = self.loss(outputs, labels)

        labels_hat = torch.argmax(outputs, dim=1)
        train_acc = torch.sum(labels.data == labels_hat).item() / (len(labels) * 1.0)

        self.eval()

        return {
            'loss': loss,
            'train_acc': train_acc
        }
    
    def training_epoch_end(self, training_step_outputs):
        self.training = True
        self.train()

        train_acc = np.mean([x['train_acc'] for x in training_step_outputs])
        train_acc = torch.tensor(train_acc, dtype=torch.float32)
        print("train_acc", train_acc)
        train_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        
        neptune.log_metric('train_loss', train_loss)
        neptune.log_metric('train acc', train_acc)
        # self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.epoch)
        self.eval()

        return {
            'log': {
                'train_loss': train_loss,
                'train_acc': train_acc
                },
            'progress_bar': {
                'train_loss': train_loss,
                'train_acc': train_acc
            }
        }
    

    def validation_step(self, batch, batch_idx):
        # self.training = False
        # self.eval()

        inputs, labels = batch

        print(inputs.shape )

        outputs = self(inputs)
        loss = self.loss(outputs, labels)


        # _, preds = torch.max(outputs, 1)
        # running_corrects += torch.sum(preds == labels.data)

        labels_hat = torch.argmax(outputs, dim=1)
        # print("labels", labels,"labels_hat",labels_hat)
        val_acc = torch.sum(labels.data == labels_hat).item() / (len(labels) * 1.0)

        self.train()
        return {
            'val_loss': loss,
            'val_acc': val_acc
        }
    
    def validation_epoch_end(self, validation_step_outputs):
        self.training = False
        self.eval()

        val_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        # val_tot = [x['val_acc'] for x in validation_step_outputs]
        # val_acc = np.mean(val_tot)
        print("HERE\n\n\n\nValidation in each step\n")
        print([x['val_acc'] for x in validation_step_outputs])

        val_acc = np.mean([x['val_acc'] for x in validation_step_outputs])
        val_acc = torch.tensor(val_acc, dtype=torch.float32)
        print("val_loss", val_loss)
        print("val_acc", val_acc)

        neptune.log_metric('val_loss', val_loss)
        neptune.log_metric('val acc', val_acc)


        self.epoch += 1
        self.train()
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
            crops.append(img) 
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
    # logger = pl_loggers.TensorBoardLogger('/lab/vislab/OPEN/justin/lightning-OPEN/logs/layer_4/')
    trainer = pl.Trainer(
        max_epochs=25,
        num_sanity_val_steps=-1,
        gpus=[2] if torch.cuda.is_available() else None,
        # logger=logger
    ) 

    model_ft = Model()
    # ct=0
    # for child in model_ft.model1.children():
    #     ct += 1
    #     if ct < 8: # freezing the first few layers to prevent overfitting
    #         for param in child.parameters():
    #             param.requires_grad = False
    # ct=0
    # for child in model_ft.model2.children():
    #     ct += 1
    #     if ct < 8:
    #         for param in child.parameters():
    #             param.requires_grad = False


    trainer.fit(model_ft)