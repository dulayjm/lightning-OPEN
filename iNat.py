from __future__ import absolute_import

import neptune

neptune.init('dulayjm/sandbox')
neptune.create_experiment(name='minimal_example', params={'lr': 0.001}, tags=['resnet', 'iNat'])

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


# train_path = '/lab/vislab/OPEN/datasets_RGB_one/train/'
# valid_path = '/lab/vislab/OPEN/datasets_RGB_one/val/'

num_classes = 682 # we need to adjust this here for the number of classes in plants 

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        """TODO set some sort of torch thing to keep tensors same size....
        """
        row = self.dataframe.iloc[index]
        return (
            torchvision.transforms.functional.to_tensor(Image.open(row['file_name'])),
            row['category_id'],
        )

def default_loader(path):
    return Image.open(path).convert('RGB')


class INAT(torch.utils.data.Dataset):
    def __init__(self, root, ann_file, is_train=True):

        # load annotations
        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        self.imgs = [aa['file_name'] for aa in ann_data['images']]
        self.ids = [aa['id'] for aa in ann_data['images']]

        # if we dont have class labels set them to '0'
        if 'annotations' in ann_data.keys():
            self.classes = [aa['category_id'] for aa in ann_data['annotations']]
        else:
            self.classes = [0]*len(self.imgs)

        # print out some stats
        print ('\t' + str(len(self.imgs)) + ' images')
        print ('\t' + str(len(set(self.classes))) + ' classes')

        self.root = root
        self.is_train = is_train
        self.loader = default_loader

        # augmentation params
        self.im_size = [299, 299]  # can change this to train on higher res
        self.mu_data = [0.485, 0.456, 0.406]
        self.std_data = [0.229, 0.224, 0.225]
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.25

        # augmentations
        self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
        self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)

    def __getitem__(self, index):
        path = self.root + self.imgs[index]
        im_id = self.ids[index]
        img = self.loader(path)
        species_id = self.classes[index]

        if self.is_train:
            img = self.scale_aug(img)
            img = self.flip_aug(img)
            img = self.color_aug(img)
        else:
            img = self.center_crop(img)

        img = self.tensor_aug(img)
        img = self.norm_aug(img)

        return img, im_id, species_id

    def __len__(self):
        return len(self.imgs)


class MetricCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class Model(LightningModule):
    """ Model 
    """

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
        self.fc=nn.Linear(4096,num_classes) # the first number is the number of weights? The second is the num_classes


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

        # ann_file = './input/train2019.json'
        # with open(ann_file) as data_file:
        #     train_anns = json.load(data_file)
        # train_anns_df = pd.DataFrame(train_anns['annotations'])[['image_id','category_id']]
        # train_img_df = pd.DataFrame(train_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
        # df_train_file_cat = pd.merge(train_img_df, train_anns_df, on='image_id')
        # df_train_file_cat['category_id']=df_train_file_cat['category_id'].astype(str)
        # self.trainset = DatasetWrapper(df_train_file_cat)


        # valid_ann_file = './input/val2019.json'
        # with open(valid_ann_file) as data_file:
        #         valid_anns = json.load(data_file)
        # valid_anns_df = pd.DataFrame(valid_anns['annotations'])[['image_id','category_id']]
        # valid_img_df = pd.DataFrame(valid_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
        # df_valid_file_cat = pd.merge(valid_img_df, valid_anns_df, on='image_id')
        # df_valid_file_cat['category_id']=df_valid_file_cat['category_id'].astype(str)
        # self.validset = DatasetWrapper(df_valid_file_cat)

        # self.trainset = INAT('./', './input/train2019.json', is_train=True)
        # self.validset = INAT('./', './input/val2019.json', is_train=False



        # self.trainset = datasets.ImageFolder(train_path, data_transforms['train'])
        # self.validset = datasets.ImageFolder(valid_path, data_transforms['valid'])

        # just plants here: 
        data_dir = '/lab/vislab/OPEN/iNaturalist/train_val2019/Plants/'

        # dataset = datasets.ImageFolder(data_dir)

        # validation_split = 0.8
        # dataset_size = len(dataset)
        # indices = list(range(dataset_size))
        # split = int(np.floor(validation_split * dataset_size))






        # load the dataset
        self.trainset = datasets.ImageFolder(
            data_dir, data_transforms['train'],
        )

        self.validset = datasets.ImageFolder(
            data_dir, data_transforms['valid'],
        )

        test_size=0.2
        num_train = len(self.trainset)
        indices = list(range(num_train))
        split = int(np.floor(test_size * num_train))
        train_idx, test_idx = indices[split:], indices[:split]

        print("train_idx", len(train_idx))
        print("test_idx", len(test_idx))

        print("\n\n\n\n\n\n\n")

        valid_size=0.2
        num_split_val = len(train_idx)
        split_val = int(np.floor(valid_size * num_split_val))
        train_idx, valid_idx = indices[split_val:], indices[:split_val]

        print(len(train_idx), len(valid_idx))

        1/0
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)
        self.test_sampler = SubsetRandomSampler(test_idx)


    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=25, shuffle=False, sampler=self.train_sampler, num_workers=25)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=25, shuffle=False, sampler=self.valid_sampler, num_workers=25)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        self.model1.train()
        self.model2.train()
        
        # for imgs, labels in model_ft.trainset: 
        #     print(labels)

        self.training = True
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss(outputs, labels)

        labels_hat = torch.argmax(outputs, dim=1)
        train_acc = torch.sum(labels.data == labels_hat).item() / (len(labels) * 1.0)

        self.model1.eval()
        self.model2.eval()

        return {
            'loss': loss,
            'train_acc': train_acc
        }
    
    def training_epoch_end(self, training_step_outputs):
        self.training = True
        self.model1.train()
        self.model2.train()

        train_acc = np.mean([x['train_acc'] for x in training_step_outputs])
        train_acc = torch.tensor(train_acc, dtype=torch.float32)
        print("train_acc", train_acc)
        train_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        neptune.log_metric('train_loss', train_loss)
        neptune.log_metric('train acc', train_acc)

        # self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.epoch)
        self.model1.eval()
        self.model2.eval()

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
        self.model1.eval()
        self.model2.eval()

        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss(outputs, labels)


        # _, preds = torch.max(outputs, 1)
        # running_corrects += torch.sum(preds == labels.data)

        labels_hat = torch.argmax(outputs, dim=1)
        # print("labels", labels,"labels_hat",labels_hat)
        val_acc = torch.sum(labels.data == labels_hat).item() / (len(labels) * 1.0)

        self.model1.train()
        self.model2.train()
        return {
            'val_loss': loss,
            'val_acc': val_acc
        }
    
    def validation_epoch_end(self, validation_step_outputs):
        self.training = False
        self.model1.eval()
        self.model2.eval()

        val_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        # val_tot = [x['val_acc'] for x in validation_step_outputs]
        # val_acc = np.mean(val_tot)
        print("HERE\n\n\n\nValidation in each step\n")
        # print([x['val_acc'] for x in validation_step_outputs])

        val_acc = np.mean([x['val_acc'] for x in validation_step_outputs])
        val_acc = torch.tensor(val_acc, dtype=torch.float32)
        print("val_loss", val_loss)
        print("val_acc", val_acc)

        neptune.log_metric('val_loss', val_loss)
        neptune.log_metric('val acc', val_acc)


        self.epoch += 1
        self.model1.train()
        self.model2.train()
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
    # create some sort of ablation study
    logger = pl_loggers.TensorBoardLogger('/lab/vislab/OPEN/justin/lightning-OPEN/iNat_logs/v2/')
    metrics_callback = MetricCallback()
    # logger = pl_loggers.TensorBoardLogger('/lab/vislab/OPEN/justin/lightning-OPEN/logs/layer_4/')
    trainer = pl.Trainer(
        max_epochs=15,
        num_sanity_val_steps=-1,
        gpus=[1] if torch.cuda.is_available() else None,
        callbacks=[metrics_callback],
        logger=logger
    ) 

    model_ft = Model()
    ct=0
    for child in model_ft.model1.children():
        ct += 1
        if ct < 8: # freezing the first few layers to prevent overfitting
            for param in child.parameters():
                param.requires_grad = False
    ct=0
    for child in model_ft.model2.children():
        ct += 1
        if ct < 8: 
            for param in child.parameters():
                param.requires_grad = False


    trainer.fit(model_ft)