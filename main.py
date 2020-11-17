from __future__ import absolute_import

import argparse
import gc
import os
import neptune
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

from src.Model import Model
from src.Dataset import INAT
from src.Utils import *

parser = argparse.ArgumentParser(description='running parameters')
parser.add_argument('--Data', type=str, help='dataset name: inat, inat-plants, RGB_one, RGB_two')
parser.add_argument('--lr', type=float, help='initial learning rate')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--epochs', type=int, help='epochs')
parser.add_argument('--gpu', type=int, help='gpu')
parset.add_argument('--log', type=bool, help='log')
args = parser.parse_args()


###################################################  
if args.log: 
    neptune.init('dulayjm/sandbox')
    neptune.create_experiment(name='whole_iNaturalist_dataset', params={'lr': 0.0045}, tags=['resnet', 'iNat'])

# recorder frequency
num_epochs = args.epochs

trainer = pl.Trainer(
        max_epochs=num_epochs,
        num_sanity_val_steps=-1,
        gpus=[args.gpu] if torch.cuda.is_available() else None,
        callbacks=[metrics_callback],
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

###################################################  
# save model
torch.save(model.cpu().state_dict(), dst + 'model_state_dict.pth')
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))