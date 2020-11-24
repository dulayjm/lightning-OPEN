from __future__ import absolute_import

import neptune

neptune.init('dulayjm/sandbox')
neptune.create_experiment(name='plants_antialiasing', params={'lr': 0.015}, tags=['resnet152', 'iNat'])

import antialiased_cnns
from argparse import ArgumentParser
import gc
import os
import numpy as np
import optuna
# import matplotlib.pyplot as plt
import json
import pandas as pd
from PIL import Image
from pytorch_metric_learning import losses, samplers
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Callback
from pytorch_lightning import loggers as pl_loggers
from random import sample, random
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, Dataset


class MetricCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def ccop(conv_feats, threshold, bs, ncrops=1,pooling='ccop'):
    _, nf, h, w = conv_feats.size()
    conv_feats = conv_feats.view(bs, ncrops, nf, h, w).transpose(1, 2)
    
    if pooling.lower()=='avg':
        conv_feats = torch.mean(conv_feats,[2,3,4])
    
    elif pooling.lower()=='max':
        conv_feats = conv_feats.reshape(bs,nf,ncrops*h*w)
        conv_feats = torch.max(conv_feats,-1)[0]
    
    else:
        conv_feats = conv_feats.reshape(bs*nf, ncrops * h * w).transpose(0,1)
        conv_feats = torch.where(conv_feats >= torch.mean(conv_feats, 0) + threshold * torch.std(conv_feats, 0), conv_feats, torch.cuda.FloatTensor([float("nan")])).transpose(0,1)
        
        conv_feats = conv_feats.reshape(bs, nf, ncrops, h, w).transpose(1, 2)
        # take the average of the non-nan values
        conv_feats=torch.sum(torch.where(torch.isnan(conv_feats),torch.tensor(0.).to("cuda:3"),conv_feats),[3,4])/(torch.sum((~torch.isnan(conv_feats)),[3,4])+0.001)

        # average over the 64 crops
        conv_feats = conv_feats.view(bs, ncrops, -1)
        conv_feats = torch.mean(conv_feats, 1)
    
    return conv_feats

class Model(LightningModule):
    """ Model 
    """
    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.epoch = 0
        self.log = True
        self.learning_rate = 0.015
        self.batch_size=12
        self.dropout_rate = 0
        self.pooling = 'ccop'
        self.num_classes = 682
        # self.dataset = 'OPEN_small'
        self.multi_network = False

        # self.train_path = data[self.dataset]['train_path']
        # self.valid_path = data[self.dataset]['val_path']
        # self.means = data[self.dataset]['mean']
        # self.stds = data[self.dataset]['std']
        
        self.training_correct_counter = 0
        self.training = False
        
        self.loss=nn.CrossEntropyLoss()
        
        self.std_thresh = 3
        self.std_thresh2 = 0.5
                
        mod1 = antialiased_cnns.resnet50(pretrained=True, filter_size=4)
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
        if self.multi_network:
            mod2 = antialiased_cnns.resnet50(pretrained=True, filter_size=4)
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
        self.fc = nn.Linear(4096, self.num_classes)
        # self.fc1 = nn.Linear(4096,512)
        # self.fc2 = nn.Linear(512,self.num_classes)    

    def forward(self, x):
        x = x.transpose(1, 0)
        
        x0 = x[:-4].transpose(1,0) # high resolution crops
        x1 = x[-4:].transpose(1,0) # low resolution crops
                
        # high res
        bs, ncrops, c, h, w = x0.size()
        x0 = x0.contiguous().view((-1, c, h, w))
        x0 = self.model1(x0)
        if self.pooling.lower() == 'ccop':
            x0 = ccop(x0, self.std_thresh, bs, ncrops)
        elif self.pooling.lower() == 'max':
            x0 = ccop(x0, self.std_thresh, bs, ncrops,pooling='max')
        else:
            x0 = ccop(x0, self.std_thresh, bs, ncrops,pooling='avg')
        
        # low res
        bs, ncrops, c, h, w = x1.size()
        x1 = x1.contiguous().view((-1, c, h, w))
        if self.multi_network:
            x1 = self.model2(x1)
        else:
            x1 = self.model1(x1)
        
        if self.pooling.lower() == 'ccop':
            x1 = ccop(x1, self.std_thresh2, bs, ncrops)
        elif self.pooling.lower() == 'max':
            x1 = ccop(x1, self.std_thresh2, bs, ncrops,pooling='max')
        else:
            x1 = ccop(x1, self.std_thresh2, bs, ncrops,pooling='avg')
        
        x = torch.cat([x0, x1], 1)
        x[torch.isnan(x)] = 0.
        
        if self.training == True:
            x = F.dropout(x, self.dropout_rate) # we might want to play around w this value
        
        return self.fc(x.view(x.size(0), -1))
        # fc1 = self.fc1(x.view(x.size(0), -1))
        # fc2 = self.fc2(fc1)
        # s = nn.Sigmoid()
        # output = s(fc2)
        # return output
    
    def train(self):
        self.model1.train()
        # self.model2.train()
        self.fc.train()
        # self.fc1.train()
        # self.fc2.train()

    def eval(self):
        self.model1.eval()
        # self.model2.eval()
        self.fc.train()
        # self.fc1.eval()
        # self.fc2.eval()


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

        # just plants here: 
        data_dir = '/lab/vislab/OPEN/iNaturalist/train_val2019/Plants'

        master_dataset = datasets.ImageFolder(
            data_dir, data_transforms['train'],
        )

        # print("HERE")
        # print(len(master_dataset))
        # print(len(master_dataset)*.8)

        self.trainset, self.validset = torch.utils.data.random_split(master_dataset, (126770,31693))
        # self.trainset, self.validset = torch.utils.data.random_split(master_dataset, (2102,526))

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=False, num_workers=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, num_workers=self.batch_size)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
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

        if self.log:
            neptune.log_metric('train_loss', train_loss)
            neptune.log_metric('train acc', train_acc)

        # self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.epoch)
        torch.cuda.empty_cache()
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
        self.training = False
        self.eval()

        inputs, labels = batch
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
        # print([x['val_acc'] for x in validation_step_outputs])

        val_acc = np.mean([x['val_acc'] for x in validation_step_outputs])
        val_acc = torch.tensor(val_acc, dtype=torch.float32)
        print("val_loss", val_loss)
        print("val_acc", val_acc)

        if self.log: 
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
    
    def random_crops(self, img, k, s):
        crops = []
        rand = torchvision.transforms.RandomCrop(s)
        Res=torchvision.transforms.Resize(256,interpolation=2)
        for j in range(k):
            im = Res(rand(img))
            crops.append(im)
        return crops

    def crops_and_random(self, img):

        res = torchvision.transforms.Resize(1024, interpolation=2)
        img=res(img)
        #Rand = torchvision.transforms.RandomCrop(512)
        #img1 = Rand(img)
        #Res = torchvision.transforms.Resize(256, interpolation=2)
        #crop512=self.random_crops(img, 4, s=512)
        #crop64=[]
        #for c in crop512:
            #crop64.append(self.random_crops(c, 1, s=64)[0])
        return self.random_crops(img,4,128)+self.random_crops(img,4, 512)

    def val_crops(self, img):
        res = torchvision.transforms.Resize(1024, interpolation=2)
        img = res(img)
        Cent1024= torchvision.transforms.CenterCrop((1024,1024))
        Cent256 = torchvision.transforms.CenterCrop((1024,1024))
        img1=np.array(Cent1024(img))
        #im = np.array(img)
        re = torchvision.transforms.Resize((256, 256), interpolation=2)
        im=np.array(Cent256(img))

        crs128 = []
        for i in range(8):
            for j in range(8):
                crs128.append(
                    re(Image.fromarray((im[i * 128:((i + 1) * 128), j *128:((j + 1) * 128)]).astype('uint8')).convert(
                        'RGB')))

        """
        Cent64 = torchvision.transforms.CenterCrop(64)
        """

        crs512 = []
        for i in range(2):
            for j in range(2):
                crs512.append(
                    re(Image.fromarray((img1[i * 512:((i + 1) * 512), j *512:((j + 1) * 512)]).astype('uint8')).convert(
                        'RGB')))

        return crs128+crs512

    def RandomErase(self, img, p=0.5, s=(0.06, 0.12), r=(0.5, 1.5)):
        im = np.array(img)
        w, h, _ = im.shape
        S = w * h
        pi = random()
        if pi > p:
            return img
        else:
            Se = S * (random() * (s[1] - s[0]) + s[0])
            re = random() * (r[1] - r[0]) + r[0]
            He = int(np.sqrt(Se * re))
            We = int(np.sqrt(Se / re))
            if He >= h:
                He = h - 1
            if We >= w:
                We = w - 1
            xe = int(random() * (w - We))
            ye = int(random() * (h - He))
            im[xe:xe + We, ye:ye + He] = int(random() * 255)
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
    # create some sort of ablation study
    # logger = pl_loggers.TensorBoardLogger('/lab/vislab/OPEN/justin/lightning-OPEN/iNat_logs/v2/')
    metrics_callback = MetricCallback()
    # logger = pl_loggers.TensorBoardLogger('/lab/vislab/OPEN/justin/lightning-OPEN/logs/layer_4/')
    trainer = pl.Trainer(
        max_epochs=25,
        num_sanity_val_steps=2,
        gpus=[3] if torch.cuda.is_available() else None,
        callbacks=[metrics_callback],
        # logger=logger
    ) 

    model_ft = Model()
    ct=0
    for child in model_ft.model1.children():
        ct += 1
        if ct < 5: # freezing the first few layers to prevent overfitting
            for param in child.parameters():
                param.requires_grad = False
    # ct=0
    # for child in model_ft.model2.children():
    #     ct += 1
    #     if ct < 8: 
    #         for param in child.parameters():
    #             param.requires_grad = False


    trainer.fit(model_ft)