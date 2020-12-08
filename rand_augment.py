from __future__ import absolute_import

import neptune

# neptune.init('dulayjm/sandbox')
# neptune.create_experiment(name='randaugment', params={'lr': 0.001}, tags=['resnet50', 'rangAugment'])

import antialiased_cnns
from argparse import ArgumentParser
import gc
import os
import numpy as np
import optuna
import matplotlib.pyplot as plt
import json
import pandas as pd
from PIL import Image
from pytorch_metric_learning import losses, samplers
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Callback
from pytorch_lightning import loggers as pl_loggers
from random import sample, random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, Dataset

from RandAugment import RandAugment

# overhead 

class MetricCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


# let's do some tsne first 
def map_features(outputs, labels, out_file):
    # create array of column for each feature output
    feat_cols = ['feature'+str(i) for i in range(outputs.shape[1])]
    
    # make dataframe of outputs -> labels
    df = pd.DataFrame(outputs, columns=feat_cols)
    df['y'] = labels
    df['labels'] = df['y'].apply(lambda i: str(i))
    
    # clear outputs and labels
    outputs, labels = None, None
    
    # creates an array of random indices from size of outputs
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])
    
    num_examples = 10000
    
    df_subset = df.loc[rndperm[:num_examples],:].copy()
    data_subset = df_subset[feat_cols].values
    
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(data_subset)
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    
    plt.figure(figsize=(16,10))
    plt.scatter(
        x=df_subset["tsne-2d-one"],
        y=df_subset["tsne-2d-two"],
        c=df_subset["y"]
    )
    plt.savefig(out_file, bbox_inches='tight', pad_inches = 0)
    plt.close()



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
        conv_feats=torch.sum(torch.where(torch.isnan(conv_feats),torch.tensor(0.).to("cuda:1"),conv_feats),[3,4])/(torch.sum((~torch.isnan(conv_feats)),[3,4])+0.001)

        # average over the 64 crops
        conv_feats = conv_feats.view(bs, ncrops, -1)
        conv_feats = torch.mean(conv_feats, 1)
    
    return conv_feats


# class CustomDataset(torchvision.datasets.ImageFolder):

#     def __init__(self, path, transforms):
#         super(CustomDataset, self).__init__()
#         super.path = path
#         super.transforms = transforms

#     def printItem()



class Guided_backprop():
    """
        Visualize CNN activation maps with guided backprop.
        
        Returns: An image that represent what the network learnt for recognizing 
        the given image. 
        
        Methods: First layer input that minimize the error between the last layers output,
        for the given class, and the true label(=1). 
        
        ! Call visualize(image) to get the image representation
    """
    def __init__(self,model):
        self.model = model
        self.image_reconstruction = None
        self.activation_maps = []
        # eval mode
        self.model.eval()
        self.register_hooks()
    
    def register_hooks(self):
        
        def first_layer_hook_fn(module, grad_out, grad_in):
            """ Return reconstructed activation image"""
            self.image_reconstruction = grad_out[0] 
            
        def forward_hook_fn(module, input, output):
            """ Stores the forward pass outputs (activation maps)"""
            self.activation_maps.append(output)
            
        def backward_hook_fn(module, grad_out, grad_in):
            """ Output the grad of model output wrt. layer (only positive) """
            
            # Gradient of forward_output wrt. forward_input = error of activation map:
                # for relu layer: grad of zero = 0, grad of identity = 1
            grad = self.activation_maps[-1] # corresponding forward pass output 
            grad[grad>0] = 1 # grad of relu when > 0
            
            # set negative output gradient to 0 #!???
            positive_grad_out = torch.clamp(input=grad_out[0],min=0.0)
            
            # backward grad_out = grad_out * (grad of forward output wrt. forward input)
            new_grad_out = positive_grad_out * grad
            
            del self.forward_outputs[-1] 
            
            # For hook functions, the returned value will be the new grad_out
            return (new_grad_out,)
            
        # !!!!!!!!!!!!!!!! change the modules !!!!!!!!!!!!!!!!!!
        # only conv layers, no flattened fc linear layers
        modules = list(self.model._modules.items())
        
        # register hooks to relu layers
        for name, module in modules:
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_backward_hook(backward_hook_fn)
        
        # register hook to the first layer 
        first_layer = modules[0][1] 
        first_layer.register_backward_hook(first_layer_hook_fn)
        
    def visualize(self, input_image, target_class):
        # last layer output
        model_output = self.model(input_image)
        self.model.zero_grad()
        
        # only calculate gradients wrt. target class 
        # set the other classes to 0: eg. [0,0,1]
        grad_target_map = torch.zeros(model_output.shape,
                                     dtype=torch.float)
        grad_target_map[0][target_class] = 1
        
        model_output.backward(grad_target_map)
        
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        result = self.image_reconstruction.data.numpy()[0] 
        return result



class Model(LightningModule):
    """ Model 
    """
    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.epoch = 0
        self.log = True
        self.plot = True
        self.learning_rate = 0.001
        self.batch_size = 16
        self.training = False
        self.dropout_rate = 0
        self.num_classes = 10
        self.randAugParamN = 2
        self.randAugParamM = 7
        
        # self.dataset = 'OPEN_small'
        self.multi_network = False

        # self.train_path = data[self.dataset]['train_path']
        # self.valid_path = data[self.dataset]['val_path']
        # self.means = data[self.dataset]['mean']
        # self.stds = data[self.dataset]['std']
        
        self.training_correct_counter = 0
        self.loss=nn.CrossEntropyLoss()
        
        self.std_thresh = 0.5
        # self.std_thresh2 = 0.5
                
        mod1 = models.resnet50(pretrained=True)
        # model arch 
        self.trunk = nn.Sequential(
            mod1.conv1,
            mod1.bn1,
            mod1.relu,
            mod1.maxpool,

            mod1.layer1,
            mod1.layer2,
            mod1.layer3,
            mod1.layer4,
        )
        self.embedder = nn.Linear(2048, 10)
        
    # forward 
    def forward(self, x, pooling="ccop", images="multiple"):
        images = "single"
        if images=="single":
            bs, c, h, w = x.size()
        else:
            bs, ncrops, c, h, w = x.size()
            x = x.contiguous().view((-1, c, h, w))
        x = self.trunk(x)

        if images=="single":
            x = self.pooling_single(x, self.std_thresh, bs, pooling)
        else:
            x = self.pooling(x, self.std_thresh, bs, ncrops)

        x[torch.isnan(x)] = 0.

        if self.training == True:
            x = F.dropout(x, self.dropout_rate)

        return self.embedder(x.view(x.size(0), -1))

    def pooling(self, conv_feats, threshold, bs, ncrops=1):
        _, nf, h, w = conv_feats.size()
        conv_feats = conv_feats.view(bs, ncrops, nf, h, w).transpose(1, 2)

        conv_feats = conv_feats.reshape(bs*nf, ncrops * h * w).transpose(0,1)
        conv_feats = torch.where(conv_feats >= torch.mean(conv_feats, 0) + threshold * torch.std(conv_feats, 0), conv_feats, torch.cuda.FloatTensor([float("nan")])).transpose(0,1)
        conv_feats = conv_feats.reshape(bs, nf, ncrops, h, w).transpose(1, 2)
        
        # take the average of the non-nan values
        conv_feats=torch.sum(torch.where(torch.isnan(conv_feats),torch.cuda.FloatTensor([0.]),conv_feats),[3,4])/(torch.sum((~torch.isnan(conv_feats)),[3,4])+0.001)

        # average over the 64 crops
        conv_feats = conv_feats.view(bs, ncrops, -1)
        conv_feats = torch.mean(conv_feats, 1)
        return conv_feats
    
    def pooling_single(self, conv_feats, threshold, bs, pooling='ccop'):
        _, nf, h, w = conv_feats.size()

        if pooling.lower()=='avg':
            conv_feats = torch.mean(conv_feats,[2,3])

        elif pooling.lower()=='max':
            conv_feats = conv_feats.reshape(bs,nf,h*w)
            conv_feats = torch.max(conv_feats,-1)[0]

        else:
            conv_feats = conv_feats.reshape(bs*nf, h * w).transpose(0,1)
            conv_feats = torch.where(conv_feats >= torch.mean(conv_feats, 0) + threshold * torch.std(conv_feats, 0), conv_feats, torch.cuda.FloatTensor([float("nan")])).transpose(0,1)
            conv_feats = conv_feats.reshape(bs, nf, h, w)
            # take the average of the non-nan values
            conv_feats = torch.sum(torch.where(torch.isnan(conv_feats),torch.cuda.FloatTensor([0.]),conv_feats),[2,3])/(torch.sum((~torch.isnan(conv_feats)),[2,3])+0.001)
        return conv_feats

    def train(self):
        self.trunk.train()
        # self.model2.train()
        self.embedder.train()
        # self.fc1.train()
        # self.fc2.train()

    def eval(self):
        self.trunk.eval()
        # self.model2.eval()
        self.embedder.eval()
        # self.fc1.eval()
        # self.fc2.eval()


    def prepare_data(self):
        
        ##### RandAugment changes here for now

        data_transforms = { # same for now
            'train': transforms.Compose([
            transforms.Resize((256, 256), interpolation=2),

#             transforms.RandomCrop(32, padding=0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[n / 255.
                                                                                                        for n in
                                                                                                        [129.3, 124.1,
                                                                                                        112.4]],
                                                                                                std=[n / 255. for n in
                                                                                                    [68.2, 65.4,
                                                                                                        70.4]])

            ]),
            
            'valid': transforms.Compose([
            transforms.Resize((256, 256), interpolation=2),
#             transforms.RandomCrop(32, padding=0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[n / 255.
                                                                                                        for n in
                                                                                                        [129.3, 124.1,
                                                                                                        112.4]],
                                                                                                std=[n / 255. for n in
                                                                                                    [68.2, 65.4,
                                                                                                        70.4]])

            ]),
        }
        data_transforms['train'].transforms.insert(0, RandAugment(self.randAugParamN, self.randAugParamM))

        train_path = '/lab/vislab/OPEN/datasets_RGB_new/train/'
        valid_path = '/lab/vislab/OPEN/datasets_RGB_new/val/'

        self.trainset = torchvision.datasets.ImageFolder(train_path, data_transforms['train'])
        self.validset = torchvision.datasets.ImageFolder(valid_path, data_transforms['valid'])
        

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.batch_size, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, num_workers=self.batch_size, drop_last=True)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        self.train()
        self.training = True
        
        inputs, labels = batch
        outputs = self(inputs, pooling='ccop')
        loss = self.loss(outputs, labels)

        labels_hat = torch.argmax(outputs, dim=1)
        train_acc = torch.sum(labels.data == labels_hat).item() / (len(labels) * 1.0)


        
        if self.plot: 
            # yeah something with this
            # then save it 
            idx = torch.randint(0, outputs.size(0), ())
            img = inputs[idx, 0]
            # inv_normalize = transforms.Normalize(
            #     mean=[-68.2/129.3, -65.4/124.1, -70.4/112.4],
            #     std=[1/129.3, 1/124.1, 1/112.4]
            # )
            plt.figure()
            plt.imshow(img.detach().cpu().numpy())
            plt.savefig('sample-{}.png'.format(self.epoch))
            plt.close()
            self.plot = False
            # self.visualizer = Guided_backprop(self.trunk)

        # print("HERE: visualize")
        # img = inputs[0]
        # lab = labels[0]
        # print("img.shape", img.shape)
        # print("lab.shape,", lab.shape)
        # result = self.visualizer.visualize(img, lab)
        # print(type(result))

        # del self.visualizer

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

        # if self.log:
        #     neptune.log_metric('train_loss', train_loss)
        #     neptune.log_metric('train acc', train_acc)

        self.eval()
        torch.cuda.empty_cache()


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
        outputs = self(inputs, pooling='ccop')
        loss = self.loss(outputs, labels)
        
        labels_hat = torch.argmax(outputs, dim=1)
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
        # print("HERE\n\n\n\nValidation in each step\n")
        # print([x['val_acc'] for x in validation_step_outputs])

        val_acc = np.mean([x['val_acc'] for x in validation_step_outputs])
        val_acc = torch.tensor(val_acc, dtype=torch.float32)
        print("val_loss", val_loss)
        print("val_acc", val_acc)

        # if self.log: 
        #     neptune.log_metric('val_loss', val_loss)
        #     neptune.log_metric('val acc', val_acc)

        
        self.epoch += 1
        self.plot = True
        self.train()
        torch.cuda.empty_cache()

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
    

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.1)
        return parser


# In[22]:


if __name__ == '__main__':
    # create some sort of ablation study
    # logger = pl_loggers.TensorBoardLogger('/lab/vislab/OPEN/justin/lightning-OPEN/iNat_logs/v2/')
    metrics_callback = MetricCallback()
    # logger = pl_loggers.TensorBoardLogger('/lab/vislab/OPEN/justin/lightning-OPEN/logs/layer_4/')
    trainer = pl.Trainer(
        max_epochs=50,
        num_sanity_val_steps=-1,
        gpus=[1] if torch.cuda.is_available() else None,
        callbacks=[metrics_callback],
        # logger=logger
        auto_lr_find=True,
    ) 

    model_ft = Model()
    ct=0
    for child in model_ft.trunk.children():
        ct += 1
        if ct < 8: # freezing the first few layers to prevent overfitting
            for param in child.parameters():
                param.requires_grad = False
    # ct=0
    # for child in model_ft.model2.children():
    #     ct += 1
    #     if ct < 8: 
    #         for param in child.parameters():
    #             param.requires_grad = False


    trainer.fit(model_ft)