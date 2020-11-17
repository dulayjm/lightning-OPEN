from __future__ import absolute_import

import neptune

neptune.init('dulayjm/sandbox')
neptune.create_experiment(name='whole_iNaturalist_dataset_new_model', params={'lr': 0.0045}, tags=['resnet', 'iNat'])

from argparse import ArgumentParser
import json
import os
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Callback
from pytorch_lightning import loggers as pl_loggers
from random import random
import torch
import time
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models


# train_path = '/lab/vislab/DATA/SUN397/SUN_10/train/'
# valid_path = '/lab/vislab/DATA/SUN397/SUN_10/val/'

num_classes = 1010

class MetricCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

################################################################
# iNaturalist imports

def random_crops(img, k, s):
    crops = []
    rand = torchvision.transforms.RandomCrop(s)
    Res=torchvision.transforms.Resize(256,interpolation=2)
    for j in range(k):
        im = Res(rand(img))
        crops.append(im)
    return crops

def crops_and_random(img):

    res = torchvision.transforms.Resize(1024, interpolation=2)
    img=res(img)
    #Rand = torchvision.transforms.RandomCrop(512)
    #img1 = Rand(img)
    #Res = torchvision.transforms.Resize(256, interpolation=2)
    #crop512=self.random_crops(img, 4, s=512)
    #crop64=[]
    #for c in crop512:
        #crop64.append(self.random_crops(c, 1, s=64)[0])
    return random_crops(img,4,128)+random_crops(img,4, 512)

def val_crops(img):
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

def RandomErase(img, p=0.5, s=(0.06, 0.12), r=(0.5, 1.5)):
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



def default_loader(path):
    return Image.open(path).convert('RGB')

def load_taxonomy(ann_data, tax_levels, classes):
    # loads the taxonomy data and converts to ints
    taxonomy = {}

    if 'categories' in ann_data.keys():
        num_classes = len(ann_data['categories'])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data['categories']]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0]*len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    print(classes_taxonomic)
    print(taxonomy ,'\n\n\n\n\n\n')

    return taxonomy, classes_taxonomic


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

        # load taxonomy
        self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
                           #8142, 4412,    1120,     273,     57,      25,       6
        self.taxonomy, self.classes_taxonomic = load_taxonomy(ann_data, self.tax_levels, self.classes)

        # print out some stats
        print( '\t' + str(len(self.imgs)) + ' images' )
        print( '\t' + str(len(set(self.classes))) + ' classes' )

        self.root = root
        self.is_train = is_train
        self.loader = default_loader

        # # augmentation params
        self.im_size = [256, 256]  # can change this to train on higher res
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

        self.train_compose = transforms.Compose([
            transforms.Lambda(lambda img: RandomErase(img)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Lambda(lambda img: crops_and_random(img)),
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

            ])

        self.valid_compose = transforms.Compose([
            transforms.Lambda(lambda img: val_crops(img)),
            transforms.Lambda(lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Normalize(mean=[n / 255.
                                                                                                        for n in
                                                                                                        [129.3, 124.1,
                                                                                                        112.4]],
                                                                                                std=[n / 255. for n in
                                                                                                    [68.2, 65.4,
                                                                                                        70.4]])])(crop) for
                                                        crop in crops]))

            ])

    def __getitem__(self, index):
        path = self.root + self.imgs[index]
        im_id = self.ids[index]
        img = self.loader(path)
        species_id = self.classes[index]
        tax_ids = self.classes_taxonomic[species_id]

        if self.is_train:
            # img = self.scale_aug(img)
            # img = self.flip_aug(img)
            # img = self.color_aug(img)
            img = self.train_compose(img)

        else:
            img = self.valid_compose(img)


        # print("end of __getitem__")
        # print("type(img)", type(img))
        # print("img", img)
        # img = self.tensor_aug(img)
        # img = self.norm_aug(img)

        # return img, im_id, species_id, tax_ids
        # img = torch.tensor(img)
        return img, species_id


    def __len__(self):
        return len(self.imgs)
###############################################################

class Model(LightningModule):
    """ Model
    """

    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.epoch = 0
        self.learning_rate = 0.0045
        self.training_correct_counter = 0
        self.training = False
        self.batch_size=16
        self.loss=nn.CrossEntropyLoss()

        mod1 = models.resnet50(pretrained=True)
        #mod2 = models.resnet50(pretrained=True)
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
        self.fc = nn.Linear(4096, num_classes)
    def forward(self, x):
        x = x.transpose(1, 0)
        #nc,_,_,_,_=x.size()
        x0 = x[:-4].transpose(1,0)
        x1 = x[-4:].transpose(1,0)
        bs, ncrops, c, h, w = x0.size()
        #bs,c, h, w = x0.size()
        x0 = x0.contiguous().view((-1, c, h, w))
        x0 = self.model1(x0)
        #x0 = F.avg_pool2d(x0, 8)
        _, nf, h, w = x0.size()
        x0 = x0.view(bs, ncrops, nf, h, w).transpose(1, 2).reshape(bs * nf, ncrops * h * w).transpose(0, 1)
        x0 = torch.where(x0 >= torch.mean(x0, 0) + 4 * torch.std(x0, 0), x0, torch.tensor(float("nan")).to("cuda:2")).transpose(0,1)
        x0 = x0.reshape(bs, nf, ncrops, h, w).transpose(1, 2)
        x0=torch.sum(torch.where(torch.isnan(x0),torch.tensor(0.).to("cuda:2"),x0),[3,4])/(torch.sum((~torch.isnan(x0)),[3,4])+0.001)
        #x0 = F.avg_pool2d(x0.reshape(bs * ncrops, nf, h, w), 8)
        x0 = x0.view(bs, ncrops, -1)
        x0 = torch.mean(x0, 1)
        #x0 = x0.view(bs, -1)
        #x0, _ = torch.max(x0.view(bs, ncrops, -1), 1)
        #x0 = torch.stack([torch.sum(ax * F.softmax(ax, 0), 0) for ax in x0])
        bs, ncrops, c, h, w = x1.size()
        #bs, c, h, w = x1.size()
        x1 = x1.contiguous().view((-1, c, h, w))
        x1 = self.model2(x1)
        x1 = F.avg_pool2d(x1, 8)
        #x1= x1.view(bs,ncrops,-1)
        #x1 = torch.stack([torch.where(ax > torch.mean(ax,0) + 1, ax, torch.tensor(0.).to("cuda:0")) for ax in x1])
        #x1 = torch.stack([torch.sum(ax, 0) / (torch.sum((ax != 0.), 0) + 1) for ax in x1])
        #x1 = torch.stack([torch.sum(ax * F.softmax(ax, 0), 0) for ax in x1])
        #x1= x1.view(bs,-1)
        #x1, _ = torch.max(x1, 1)
        _, nf, h, w = x1.size()
        x1 = x1.view(bs, ncrops, nf, h, w).transpose(1, 2).reshape(bs * nf, ncrops * h * w).transpose(0, 1)
        x1 = torch.where(x1 >= torch.mean(x1, 0) + 1 * torch.std(x1, 0), x1,
                         torch.tensor(float("nan")).to("cuda:2")).transpose(0, 1)
        x1 = x1.reshape(bs, nf, ncrops, h, w).transpose(1, 2)
        x1 = torch.sum(torch.where(torch.isnan(x1), torch.tensor(0.).to("cuda:2"), x1), [3, 4]) / (
                    torch.sum((~torch.isnan(x1)), [3, 4]) + 0.001)
        # x0 = F.avg_pool2d(x0.reshape(bs * ncrops, nf, h, w), 8)
        x1 = x1.view(bs, ncrops, -1)
        x1 = torch.mean(x1, 1)
        x = torch.cat([x0, x1], 1)
        if self.training == True:
            x = F.dropout(x, 0.3)
        return self.fc(x.view(x.size(0), -1))
    def train(self):
        self.model1.train()
        self.model2.train()
        self.fc.train()
    def eval(self):
        self.model1.eval()
        self.model2.eval()
        self.fc.eval()
    def prepare_data(self):

        data_transforms = {
            'train': transforms.Compose([

                transforms.Lambda(lambda img: self.RandomErase(img)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Lambda(lambda img: self.crops_and_random(img)),
                transforms.Lambda(lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(),
                                                                                 transforms.Normalize(mean=[n / 255.
                                                                                                            for n in
                                                                                                            [75.58, 96.37, 92.88]],
                                                                                                      std=[n / 255. for
                                                                                                           n in
                                                                                                           [43.36, 53.14, 52.06]])])(
                    crop) for
                                                             crop in crops]))

            ]),

            # currently same as train
            'valid': transforms.Compose([
                transforms.Lambda(lambda img: self.val_crops(img)),
                transforms.Lambda(lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(),
                                                                                 transforms.Normalize(mean=[n / 255.
                                                                                                            for n in
                                                                                                            [75.58, 96.37, 92.88]],
                                                                                                      std=[n / 255. for
                                                                                                           n in
                                                                                                           [43.36, 53.14, 52.06]])])(
                    crop) for
                                                             crop in crops]))

            ]),
        }

        # self.trainset = datasets.ImageFolder(train_path, data_transforms['train'])
        # self.validset = datasets.ImageFolder(valid_path, data_transforms['valid'])

        data_dir = '/lab/vislab/OPEN/iNaturalist/'      
        train_file = '/lab/vislab/OPEN/iNaturalist/input/train2019.json'
        val_file = '/lab/vislab/OPEN/iNaturalist/input/val2019.json'
        # data loading code
        # from the INAT class loader pytorch github
        self.trainset = INAT(data_dir, train_file,
                        is_train=True)
        self.validset = INAT(data_dir, val_file,
                        is_train=False)


    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, sampler=None, num_workers=32)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=2, shuffle=False, sampler=None, num_workers=32)

    def configure_optimizers(self):
        optimizer=torch.optim.SGD(self.parameters(), lr=self.learning_rate,momentum=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

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

        # pick a random 50% of data for each epoch
        # we don't know the right thresholds and right learning rate
        # and find the thresholds

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

    

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.1)
        return parser


if __name__ == '__main__':
    log_dir="lightning_one_MIT"
    metrics_callback = MetricCallback()
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = pl_loggers.TensorBoardLogger(log_dir)
    trainer = pl.Trainer(
        check_val_every_n_epoch=5,
        max_epochs=25,
        num_sanity_val_steps=2,
        gpus=[1] if torch.cuda.is_available() else None,
        callbacks=[metrics_callback],
        logger=logger
    )

    model_ft = Model()
    ct = 0
    for child in model_ft.model1.children():
        ct += 1
        if ct < 5:  # freezing the first few layers to prevent overfitting
            for param in child.parameters():
                param.requires_grad = False
    """ct = 0
    for child in model_ft.model2.children():
        ct += 1
        if ct < 5:
            for param in child.parameters():
                param.requires_grad = False"""

    trainer.fit(model_ft)
    trainer.save_checkpoint(log_dir+"/model.ckpt")