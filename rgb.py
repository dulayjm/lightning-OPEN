from __future__ import absolute_import

import neptune

# neptune.init('dulayjm/sandbox')
# neptune.create_experiment(name='plants_only_imagefolder', params={'lr': 0.0045}, tags=['resnet', 'iNat'])

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
# from random import sample, random
import random 
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, Dataset


num_classes = 10 

######################################################
# 2-Map imports
import importlib
import datetime
import matplotlib.pyplot as plt

def draw(emb1,emb2,label,name='',title1='',title2='',savepath='./',savename='comparison'):
    """ Draw a comparison image for two 2-d data
    Parameters
    ----------
    emb1:array of shape (n_samples,2)
        2-d data to be drawed.
    emb2:array of shape (n_samples,2)
        2-d data to be drawed.
    label: list of shape (n_samples,)
        label for data, used to draw visualization.
    name: string(optional, default ''):
        The title of this comparison.
    title1: string(optional, default ''):
        title for first figure.
    title2: string(optional, default ''):
        title for second figure.
    savepath: string(optional, default './'):
        savepath.
    savename: string(optional, default 'comparison'):
        Savename for this comparison.
    Returns
    -------
    """

    size = len(label)
    a=label%7
    b =label%12
    markers = ["." , "," , "o" , "v" , "^" , "<", ">","x","1","2","3","4"]
    colors = ['r','g','b','c','m', 'y', 'k']
    
    colorslist3=[]
    markerslist3 = []
    for item in a:
        colorslist3.append(colors[item])
    for item in b:
        markerslist3.append(markers[item])
        
    plt.figure(figsize=(20,10))
    plt.axis((-250,250,-250,250))
    plt.title(name)
    
    axe1=plt.subplot(1,2,1)
    axe1.set_title(title1)
    
    axe2=plt.subplot(1,2,2)
    axe2.set_title(title2)
    x_max = np.max([np.max(emb1[:,0]), np.max(emb2[:, 0])])+2
    y_max = np.max([np.max(emb1[:,1]), np.max(emb2[:, 1])])+2
    x_min = np.min([np.min(emb1[:,0]), np.min(emb2[:, 0])])-2
    y_min = np.min([np.min(emb1[:,1]), np.min(emb2[:, 1])])-2
    axis_max = int(max(x_max, y_max))
    axis_min = int(min(x_min, y_min))
    for k in range(0,size-1):
        axe1.scatter(emb1[k, 0], emb1[k, 1],c=colorslist3[k],marker=markerslist3[k],alpha=0.5)
        axe1.set_xlim(axis_min, axis_max)
        axe1.set_ylim(axis_min, axis_max)
        axe2.scatter(emb2[k, 0], emb2[k, 1],c=colorslist3[k],marker=markerslist3[k],alpha=0.5)
        axe2.set_xlim(axis_min, axis_max)
        axe2.set_ylim(axis_min, axis_max)
        
    plt.savefig(savepath+'/'+savename+'.png',dpi=300)
    #plt.show()

def draw_single(emb,label,name,savepath='./',axis_size=35,savename=''):
    """ Draw a image for a 2-d data
    Parameters
    ----------
    emb:array of shape (n_samples,2)
        2-d data to be drawed.
    label: list of shape (n_samples,)
        label for data, used to draw visualization.
    name: string(optional, default ''):
        The title of this figure.
    axis_size: int(optional, default ''):
        The size of axis.
    savename: string(optional, default ''):
        Savename for this comparison.
    Returns
    -------
    """
    size = len(label)
    a=label%7
    b =label%12
    markers = ["." , "," , "o" , "v" , "^" , "<", ">","x","1","2","3","4"]
    colors = ['r','g','b','c','m', 'y', 'k']
    
    colorslist3=[]
    markerslist3 = []
    for item in a:
        colorslist3.append(colors[item])
    for item in b:
        markerslist3.append(markers[item])
        
    plt.figure(figsize=(10,10))
    plt.axis((-250,250,-250,250))
    plt.title(name)
    
    axe1=plt.subplot(1,1,1)
    
    for k in range(0,size-1):
        axe1.scatter(emb[k, 0], emb[k, 1],c=colorslist3[k],marker=markerslist3[k],alpha=0.5)
        axe1.set_xlim(-axis_size, axis_size)
        axe1.set_ylim(-axis_size, axis_size)
        
    plt.savefig(savepath+'/'+savename+'.png',dpi=300)

def draw_curve(oriloss,Uloss,name='curve',savepath='./'):
    """ Draw accepted range and loss curve with different penalty scale.
    Parameters
    ----------
    oriloss:list of shape (times,)
        Loss for several times UMAP.
    Uloss: list
        loss for 2-MAP in different penalty scale.
    name: string(optional, default 'curve'):
        The title and savename of this figure.
    savepath: string(optional, default './'):
        Savepath for this curve figure.
    Returns
    -------
    """
    length = len(Uloss)
    x = range(-len(Uloss),0)
    a = [-len(Uloss),-1]
    mean = np.asarray(oriloss).mean()
    std = np.asarray(oriloss).std()
    Uloss.reverse()
    mean_line = [mean,mean]
    std_1_line = [mean+std,mean+std]
    std_2_line = [mean-std,mean-std]
    plt.figure(figsize=(8,4))
    plt.title(name)
    l1,=plt.plot(x,Uloss,linewidth=1)
    l2,=plt.plot(a,mean_line,c='r',linewidth=1)
    l3,=plt.plot(a,std_1_line,c='g',linewidth=1)
    l4,=plt.plot(a,std_2_line,c='g',linewidth=1)
    #c1,=plt.plot(x,yoked_KL_comp,c='b',linewidth=1)
    plt.legend(handles=[l1,l2,l3,l4], labels=['TUMAP cost','mean','mean+std','mean-std'],  loc='best')
    plt.xlabel("log(alpha)")
    plt.ylabel("Cost") 
    plt.savefig(savepath+name+'.png')
    plt.show()

import pickle

def yoke_TUMAP(data1,data2,label,metric='euclidean',init_1="spectral",init_2="spectral",fixed=False,n_epoches=500,times=10,name1='embed1',name2='embed2',savepath='./',all_process=False,if_draw=True):
    """2-UMAP processes including parameter selection. Running several times UMAP to get
    accepted range and search for the accepted penalty scale.
    Parameters
    ----------
    data1:array of shape (n_samples,n_vector)
        high-dimensional data which will be visualized.
    data2:array of shape (n_samples,n_vector)
        high-dimensional data which will be visualized or low-dimensional fix map if fixed is setted as True.
    label: list of shape (n_samples,)
        label for data, used to draw visualization.
    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.
    init_1: string (optional, default 'spectral')
        How to initialize the low dimensional embedding of data1. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.
    init_2: string (optional, default 'spectral')
        How to initialize the low dimensional embedding of data2. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.
    fixed: Boolean (optional, default False):
        A parameter to decide whether fixed second map. If True, 
        data2 should be set as a low-dimensional fix map.
        Making data1 yoked to fix map.
    n_epochs: int (optional, default 500)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    times: int (optional, default 10):
        The number of times to running original UMAP.
    name1: string(optional, default 'emb1'):
        Saving name and showing name in figure for data 1.
    name2: string(optional, default 'emb2'):
        Saving name and showing name in figure for data 2.
    savepath: string(optional, default './'):
        Saveing path to save result
    all_process: Boolean(optional, default 'False'):
        If find right parameter, whether continue generate for other small parameter.
    if_draw: Boolean(optional, default 'True'):
        Whether draw the 2-map comparison result.

    Returns
    -------
    result1: array of shape(n_samples,2)
        2-map result for data1.
    result2: array of shape(n_samples,2)
        2-map result for data2
    ori1: array of shape(n_samples,2)
        Umap result for data1
    ori2: array of shape(n_samples,2)
        Umap result for data2
    """

    oriloss1=[]
    oriloss2=[]
    
    ori1 = None
    ori2 = None
    for i in range(0,times):
        umaper = tumap.UMAP(metric=metric,n_epochs=n_epoches,init_1=init_1,init_2=init_2,lam=0)
        embed1,embed2 = umaper.yoke_transform(data1,data2,fixed=fixed)
        if not os.path.isdir(savepath+'/ori/'):
            os.makedirs(savepath+'/ori/')
        pickle.dump( embed1, open( savepath+'/ori/'+name1+'_'+str(i)+'.pkl', "wb" ) )
        pickle.dump( embed2, open( savepath+'/ori/'+name2+'_'+str(i)+'.pkl', "wb" ) )
        loss1,loss2 = umaper.get_semi_loss()
        oriloss1.append(loss1)
        oriloss2.append(loss2)
        
    pickle.dump( oriloss1, open( savepath+'/ori/loss1.pkl', "wb" ) )
    pickle.dump( oriloss2, open( savepath+'/ori/loss2.pkl', "wb" ) )        
    ori1 = embed1
    ori2 = embed2

    mean1 = np.asarray(oriloss1).mean()
    std1 = np.asarray(oriloss1).std() 
    max1 = np.asarray(oriloss1).max() 
    mean2 = np.asarray(oriloss2).mean()
    std2 = np.asarray(oriloss2).std()
    max2 = np.asarray(oriloss2).max() 
    
    result1 = embed1
    result2 = embed2
    Uloss1=[]
    Uloss2=[]
    for i in range(1,10):
        umaper = tumap.UMAP(metric=metric,n_epochs=n_epoches,init_1=init_1,init_2=init_2,lam=10**-i)
        embed1, embed2 = umaper.yoke_transform(data1,data2,fixed=fixed)
        if not os.path.isdir(savepath+'/yoked/'):
            os.makedirs(savepath+'/yoked/')
        pickle.dump( embed1, open( savepath+'/yoked/'+name1+'_'+str(-i)+'.pkl', "wb" ) )
        pickle.dump( embed2, open( savepath+'/yoked/'+name2+'_'+str(-i)+'.pkl', "wb" ) )
        loss1,loss2 = umaper.get_semi_loss()
        Uloss1.append(loss1)
        Uloss2.append(loss2)
        condition1 = (loss1+loss2<=mean1+std1+mean2+std2)
        if not all_process:
            if fixed is False:
                if condition1:
                    res1 = embed1
                    res2 = embed2
                    result1 = res1
                    result2 = res2
                    break
            else:
                if loss1<=mean1+std1:
                    res1 = embed1
                    res2 = embed2
                    result1 = res1
                    result2 = res2
                    break
        else:
            if fixed is False:
                if condition1:
                    res1 = embed1
                    res2 = embed2
                    if result1 is None:
                        result1 = res1
                        result2 = res2
            else:
                if loss1<=mean1+std1:
                    result1 = embed1
                    result2 = embed2
                    if result1 is None:
                        result1 = res1
                        result2 = res2
    pickle.dump(oriloss1, open( savepath+'/yoked/loss1.pkl', "wb" ) )
    pickle.dump(oriloss2, open( savepath+'/yoked/loss2.pkl', "wb" ) )
    oriloss=list(np.asarray(oriloss1)+np.asarray(oriloss2))
    Uloss=list(np.asarray(Uloss1)+np.asarray(Uloss2))
    Drawer.draw_curve(oriloss,Uloss,name='loss_curve',savepath=savepath)
    if if_draw:
        Drawer.draw(result1,result2,label,'',name1,name2,savepath=savepath,savename='comparison')
    return result1,result2,ori1,ori2

def ThruMap(datalist,label,metric='euclidean',n_epoches=500,times=5,savepath='./',if_draw=True):
    """ThruMAP processes including parameter selection. To visulize a series of datas. Running 2-map in order,
    fixing previous 2-map result, and align new one to previous one. Running several times UMAP to get 
    accepted range and search for the accepted penalty scale.
    Parameters
    ----------
    datalist:list of array, shape:(n_samples,n_vector,n)
        high-dimensional data list which will be visualized.
    label: list of shape (n_samples,)
        label for data, used to draw visualization.
    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.
    n_epochs: int (optional, default 500)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    times: int (optional, default 10):
        The number of times to running original UMAP.
    savepath: string(optional, default './'):
        Saveing path to save result
    if_draw: Boolean(optional, default 'True'):
        Whether draw the 2-map comparison result.

    Returns
    -------
    result1: array of shape(n_samples,2)
        2-map result for data1.
    result2: array of shape(n_samples,2)
        2-map result for data2
    ori1: array of shape(n_samples,2)
        Umap result for data1
    ori2: array of shape(n_samples,2)
        Umap result for data2
    """    
    data1 = datalist[0]
    data2 = datalist[1]
    
    oriloss1=[]
    oriloss2=[]
    
    ori1 = None
    ori2 = None
    
    for i in range(0,times):
        umaper = tumap.UMAP(metric=metric,n_epochs=n_epoches,lam=0)
        embed1,embed2 = umaper.yoke_transform(data1,data2,fixed=False)
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        loss1,loss2 = umaper.get_semi_loss()
        oriloss1.append(loss1)
        oriloss2.append(loss2)

    mean1 = np.asarray(oriloss1).mean()
    std1 = np.asarray(oriloss1).std() 
    max1 = np.asarray(oriloss1).max() 
    mean2 = np.asarray(oriloss2).mean()
    std2 = np.asarray(oriloss2).std()
    max2 = np.asarray(oriloss2).max() 
    
    result1 = embed1
    result2 = embed2
    for i in range(1,10):
        umaper = tumap.UMAP(metric=metric,n_epochs=n_epoches,lam=10**-i)
        embed1, embed2 = umaper.yoke_transform(data1,data2,fixed=False)
        loss1,loss2 = umaper.get_semi_loss()
        condition1 = (loss1+loss2<=mean1+std1+mean2+std2)
        if condition1:
            res1 = embed1
            res2 = embed2
            result1 = res1
            result2 = res2
            break
    pickle.dump(result1, open(savepath+'/0.pkl', "wb" ) )
    pickle.dump(result2, open(savepath+'/1.pkl', "wb" ) )
    if if_draw:
        Drawer.draw_single(result1,label,'',savepath=savepath,savename='0')
        Drawer.draw_single(result2,label,'',savepath=savepath,savename='1')
        
    fixmap=result2
    for index in range(2,len(datalist)):
        data1 = datalist[index]
        for i in range(0,times):
            umaper = tumap.UMAP(metric=metric,n_epochs=n_epoches,init_1=fixmap,init_2=fixmap,lam=0)
            embed1,_ = umaper.yoke_transform(data1,fixmap,fixed=True)
            loss1,_ = umaper.get_semi_loss()
            oriloss1.append(loss1)
            mean1 = np.asarray(oriloss1).mean()
            std1 = np.asarray(oriloss1).std() 

        result1 = embed1

        for i in range(1,10):
            umaper = tumap.UMAP(metric=metric,n_epochs=n_epoches,init_1=fixmap,init_2=fixmap,lam=10**-i)
            embed1,_= umaper.yoke_transform(data1,fixmap,fixed=True)

            loss1,_ = umaper.get_semi_loss()
            condition1 = (loss1<=mean1+std1)
            if condition1:
                result1 = embed1
                break
        pickle.dump(result1, open(savepath+'/'+str(index)+'.pkl', "wb" ) )
        fixmap=result1    
        if if_draw:
            Drawer.draw_single(result1,label,'',savepath=savepath,savename=str(index))
        
    return result1,result2,ori1,ori2

from warnings import warn
import time
import pdb
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.neighbors import KDTree

from sklearn.externals import joblib

from numpy import inf
import scipy.sparse
import scipy.sparse.csgraph
import numba

import umap.distances as dist

import umap.sparse as sparse

from umap.utils import tau_rand_int, deheap_sort, submatrix
#from umap.utils import ts
from umap.rp_tree import rptree_leaf_array, make_forest
from umap.nndescent import (
    make_nn_descent,
    make_initialisations,
    make_initialized_nnd_search,
    initialise_search,
)
from umap.spectral import spectral_layout

import locale

locale.setlocale(locale.LC_NUMERIC, "C")

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf

@numba.njit(fastmath=True)
def compute_pairwise_distances(X):
    dists = -2 * np.dot(X,X.T) + np.sum(X**2,    axis=1) + np.sum(X**2, axis=1).reshape(-1, 1)
    np.fill_diagonal(dists,0)
    dists = np.abs(dists)
    dists = np.sqrt(dists)
    #dists = np.nan_to_num(dists)
    return dists

@numba.njit(fastmath=True)
def compute_numba(embedding1,embedding2,V1,V2,a=1.577,b=0.895):
    p_d1 = compute_pairwise_distances(embedding1)
    W1 = 1/(1+a*np.power(p_d1,2*b))
    #V1 = graph1.toarray()
    
    p_d2 = compute_pairwise_distances(embedding2)
    W2 = 1/(1+a*np.power(p_d2,2*b))
    #V2 = graph2.toarray()

    R1 = (1-V1)*(np.log(1-V1)-np.log(1-W1))
    R2 = (1-V2)*(np.log(1-V2)-np.log(1-W2))
    A1 = V1*(np.log(V1)-np.log(W1))
    A2 = V2*(np.log(V2)-np.log(W2))
    return R1,R2,A1,A2

#@numba.njit(fastmath=True)
def compute_yoke_loss_semi(embedding1,embedding2,V1,V2,a=1.577,b=0.895):
    
    R1,R2,A1,A2 = compute_numba(embedding1,embedding2,V1,V2,a,b)
    #pdb.set_trace()
    R1[R1==inf]=0
    R1[R1==-inf]=0
    R1[np.logical_and(V1==0,V2==0)]=0
    #R1 = np.nan_to_num(R1)
    np.fill_diagonal(R1, 0)
    R_Loss1 = np.nansum(R1)

    #pdb.set_trace()
    R2[R2==inf]=0
    R2[R2==-inf]=0
    R2[np.logical_and(V1==0,V2==0)]=0
    #R2 = np.nan_to_num(R2)
    np.fill_diagonal(R2, 0)
    R_Loss2 = np.nansum(R2)
    
    #pdb.set_trace()
    A1[A1==inf]=0
    A1[A1==-inf]=0
    #A1 = np.nan_to_num(A1)
    np.fill_diagonal(A1, 0)
    A_Loss1 = np.nansum(A1)
    
    #pdb.set_trace()
    A2[A2==inf]=0
    A2[A2==-inf]=0
    #A2 = np.nan_to_num(A2)
    np.fill_diagonal(A2, 0)
    A_Loss2 = np.nansum(A2)
    
    Loss1 = R_Loss1+A_Loss1
    Loss2 = R_Loss2+A_Loss2
    
    return Loss1,Loss2
        
#@numba.njit(fastmath=True)
def cal_loss(embedding,graph,a=1.577,b=0.895):
    
    p_d = compute_pairwise_distances(embedding)
    W = 1/(1+a*np.power(p_d,2*b))
    V = graph.toarray()

    R = (1-V)*(np.log(1-V)-np.log(1-W))
    #pdb.set_trace()
    R[R==inf]=0
    R[R==-inf]=0
    R = np.nan_to_num(R)
    np.fill_diagonal(R, 0)
    R_Loss = R.sum()

    A = V*(np.log(V)-np.log(W))
    #pdb.set_trace()
    A[A==inf]=0
    A[A==-inf]=0
    A = np.nan_to_num(A)
    np.fill_diagonal(A, 0)
    A_Loss = A.sum()
    

    Loss = A_Loss+R_Loss
    return Loss


@numba.njit(fastmath=True) # benchmarking `parallel=True` shows it to *decrease* performance
def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In esscence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.
    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each samples. Each row should be a
        sorted list of distances to a given samples nearest neighbors.
    k: float
        The number of nearest neighbors to approximate for.
    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.
    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.
    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.
    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0])
    result = np.zeros(distances.shape[0])

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (non_zero_dists[index] - non_zero_dists[index - 1])
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0


            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances

    return result, rho


def nearest_neighbors(
    X, n_neighbors, metric, metric_kwds, angular, random_state, verbose=False
):
    """Compute the ``n_neighbors`` nearest points for each data point in ``X``
    under ``metric``. This may be exact, but more likely is approximated via
    nearest neighbor descent.
    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The input data to compute the k-neighbor graph of.
    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.
    metric: string or callable
        The metric to use for the computation.
    metric_kwds: dict
        Any arguments to pass to the metric computation function.
    angular: bool
        Whether to use angular rp trees in NN approximation.
    random_state: np.random state
        The random state to use for approximate NN computations.
    verbose: bool
        Whether to print status data during the computation.
    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.
    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.
    """
    if verbose:
        print(ts(), "Finding Nearest Neighbors")

    if metric == "precomputed":
        # Note that this does not support sparse distance matrices yet ...
        # Compute indices of n nearest neighbors
        knn_indices = np.argsort(X)[:, :n_neighbors]
        # Compute the nearest neighbor distances
        #   (equivalent to np.sort(X)[:,:n_neighbors])
        knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

        rp_forest = []
    else:
        if callable(metric):
            distance_func = metric
        elif metric in dist.named_distances:
            distance_func = dist.named_distances[metric]
        else:
            raise ValueError("Metric is neither callable, " + "nor a recognised string")

        if metric in ("cosine", "correlation", "dice", "jaccard"):
            angular = True

        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        if scipy.sparse.isspmatrix_csr(X):
            if metric in sparse.sparse_named_distances:
                distance_func = sparse.sparse_named_distances[metric]
                if metric in sparse.sparse_need_n_features:
                    metric_kwds["n_features"] = X.shape[1]
            else:
                raise ValueError(
                    "Metric {} not supported for sparse " + "data".format(metric)
                )
            metric_nn_descent = sparse.make_sparse_nn_descent(
                distance_func, tuple(metric_kwds.values())
            )

            # TODO: Hacked values for now
            n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
            n_iters = max(5, int(round(np.log2(X.shape[0]))))
            if verbose:
                print(ts(), "Building RP forest with",  str(n_trees), "trees")

            rp_forest = make_forest(X, n_neighbors, n_trees, rng_state, angular)
            leaf_array = rptree_leaf_array(rp_forest)

            if verbose:
                print(ts(), "NN descent for", str(n_iters), "iterations")
            knn_indices, knn_dists = metric_nn_descent(
                X.indices,
                X.indptr,
                X.data,
                X.shape[0],
                n_neighbors,
                rng_state,
                max_candidates=60,
                rp_tree_init=True,
                leaf_array=leaf_array,
                n_iters=n_iters,
                verbose=verbose,
            )
        else:
            metric_nn_descent = make_nn_descent(
                distance_func, tuple(metric_kwds.values())
            )
            # TODO: Hacked values for now
            n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
            n_iters = max(5, int(round(np.log2(X.shape[0]))))

            if verbose:
                print(ts(), "Building RP forest with", str(n_trees), "trees")
            rp_forest = make_forest(X, n_neighbors, n_trees, rng_state, angular)
            leaf_array = rptree_leaf_array(rp_forest)
            if verbose:
                print(ts(), "NN descent for", str(n_iters), "iterations")
            knn_indices, knn_dists = metric_nn_descent(
                X,
                n_neighbors,
                rng_state,
                max_candidates=60,
                rp_tree_init=True,
                leaf_array=leaf_array,
                n_iters=n_iters,
                verbose=verbose,
            )

        if np.any(knn_indices < 0):
            warn(
                "Failed to correctly find n_neighbors for some samples."
                "Results may be less than ideal. Try re-running with"
                "different parameters."
            )
    if verbose:
        print(ts(), "Finished Nearest Neighbor Search")
    return knn_indices, knn_dists, rp_forest


@numba.njit(parallel=True, fastmath=True)
def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    """Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.
    Parameters
    ----------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.
    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.
    sigmas: array of shape(n_samples)
        The normalization factor derived from the metric tensor approximation.
    rhos: array of shape(n_samples)
        The local connectivity adjustment.
    Returns
    -------
    rows: array of shape (n_samples * n_neighbors)
        Row data for the resulting sparse matrix (coo format)
    cols: array of shape (n_samples * n_neighbors)
        Column data for the resulting sparse matrix (coo format)
    vals: array of shape (n_samples * n_neighbors)
        Entries for the resulting sparse matrix (coo format)
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_samples * n_neighbors), dtype=np.float64)

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals


@numba.jit()
def fuzzy_simplicial_set(
    X,
    n_neighbors,
    random_state,
    metric,
    metric_kwds={},
    knn_indices=None,
    knn_dists=None,
    angular=False,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    verbose=False,
):
    """Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The data to be modelled as a fuzzy simplicial set.
    n_neighbors: int
        The number of neighbors to use to approximate geodesic distance.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.
    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean (or l2)
            * manhattan (or l1)
            * cityblock
            * braycurtis
            * canberra
            * chebyshev
            * correlation
            * cosine
            * dice
            * hamming
            * jaccard
            * kulsinski
            * mahalanobis
            * matching
            * minkowski
            * rogerstanimoto
            * russellrao
            * seuclidean
            * sokalmichener
            * sokalsneath
            * sqeuclidean
            * yule
            * wminkowski
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.
    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.
    knn_indices: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the indices of the k-nearest neighbors as a row for
        each data point.
    knn_dists: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the distances of the k-nearest neighbors as a row for
        each data point.
    angular: bool (optional, default False)
        Whether to use angular/cosine distance for the random projection
        forest for seeding NN-descent to determine approximate nearest
        neighbors.
    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.
    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    Returns
    -------
    fuzzy_simplicial_set: coo_matrix
        A fuzzy simplicial set represented as a sparse matrix. The (i,
        j) entry of the matrix represents the membership strength of the
        1-simplex between the ith and jth sample points.
    """
    
    vectices = X.shape[1]
    
    if knn_indices is None or knn_dists is None:
        knn_indices, knn_dists, _ = nearest_neighbors(
            X, n_neighbors, metric, metric_kwds, angular, random_state, verbose=verbose
        )

    sigmas, rhos = smooth_knn_dist(
        knn_dists, n_neighbors, local_connectivity=local_connectivity
    )

    rows, cols, vals = compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos
    )
    
    #new_rows = np.ones((vectices,), dtype=int)
    #new_cols = np.arange(vectices)
    #new_vals = 0.1*np.ones(vectices)
    
    #rows=np.append(rows,new_rows)
    #cols=np.append(cols,new_cols)
    #vals=np.append(vals,new_vals)
    
    #new_rows = np.arange(vectices)
    #new_cols = np.ones((vectices,), dtype=int)
    #new_vals = 0.1*np.ones(vectices)
    
    result = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    )
    result.eliminate_zeros()

    transpose = result.transpose()

    prod_matrix = result.multiply(transpose)

    result = (
        set_op_mix_ratio * (result + transpose - prod_matrix)
        + (1.0 - set_op_mix_ratio) * prod_matrix
    )

    result.eliminate_zeros()

    return result


@numba.jit()
def fast_intersection(rows, cols, values, target, unknown_dist=1.0, far_dist=5.0):
    """Under the assumption of categorical distance for the intersecting
    simplicial set perform a fast intersection.
    Parameters
    ----------
    rows: array
        An array of the row of each non-zero in the sparse matrix
        representation.
    cols: array
        An array of the column of each non-zero in the sparse matrix
        representation.
    values: array
        An array of the value of each non-zero in the sparse matrix
        representation.
    target: array of shape (n_samples)
        The categorical labels to use in the intersection.
    unknown_dist: float (optional, default 1.0)
        The distance an unknown label (-1) is assumed to be from any point.
    far_dist float (optional, default 5.0)
        The distance between unmatched labels.
    Returns
    -------
    None
    """
    for nz in range(rows.shape[0]):
        i = rows[nz]
        j = cols[nz]
        if target[i] == -1 or target[j] == -1:
            values[nz] *= np.exp(-unknown_dist)
        elif target[i] != target[j]:
            values[nz] *= np.exp(-far_dist)

    return


@numba.jit()
def reset_local_connectivity(simplicial_set):
    """Reset the local connectivity requirement -- each data sample should
    have complete confidence in at least one 1-simplex in the simplicial set.
    We can enforce this by locally rescaling confidences, and then remerging the
    different local simplicial sets together.
    Parameters
    ----------
    simplicial_set: sparse matrix
        The simplicial set for which to recalculate with respect to local
        connectivity.
    Returns
    -------
    simplicial_set: sparse_matrix
        The recalculated simplicial set, now with the local connectivity
        assumption restored.
    """
    simplicial_set = normalize(simplicial_set, norm="max")
    transpose = simplicial_set.transpose()
    prod_matrix = simplicial_set.multiply(transpose)
    simplicial_set = simplicial_set + transpose - prod_matrix
    simplicial_set.eliminate_zeros()

    return simplicial_set


@numba.jit()
def categorical_simplicial_set_intersection(
    simplicial_set, target, unknown_dist=1.0, far_dist=5.0
):
    """Combine a fuzzy simplicial set with another fuzzy simplicial set
    generated from categorical data using categorical distances. The target
    data is assumed to be categorical label data (a vector of labels),
    and this will update the fuzzy simplicial set to respect that label data.
    TODO: optional category cardinality based weighting of distance
    Parameters
    ----------
    simplicial_set: sparse matrix
        The input fuzzy simplicial set.
    target: array of shape (n_samples)
        The categorical labels to use in the intersection.
    unknown_dist: float (optional, default 1.0)
        The distance an unknown label (-1) is assumed to be from any point.
    far_dist float (optional, default 5.0)
        The distance between unmatched labels.
    Returns
    -------
    simplicial_set: sparse matrix
        The resulting intersected fuzzy simplicial set.
    """
    simplicial_set = simplicial_set.tocoo()

    fast_intersection(
        simplicial_set.row,
        simplicial_set.col,
        simplicial_set.data,
        target,
        unknown_dist,
        far_dist,
    )

    simplicial_set.eliminate_zeros()

    return reset_local_connectivity(simplicial_set)


@numba.jit()
def general_simplicial_set_intersection(simplicial_set1, simplicial_set2, weight):

    result = (simplicial_set1 + simplicial_set2).tocoo()
    left = simplicial_set1.tocsr()
    right = simplicial_set2.tocsr()

    sparse.general_sset_intersection(
        left.indptr,
        left.indices,
        left.data,
        right.indptr,
        right.indices,
        right.data,
        result.row,
        result.col,
        result.data,
        weight,
    )

    return result


@numba.jit()
def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.
    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights ofhow much we wish to sample each 1-simplex.
    n_epochs: int
        The total number of epochs we want to train for.
    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result

@numba.jit()
def merge_epochs_per_sampler(weights_0,weights_1, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.
    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights ofhow much we wish to sample each 1-simplex.
    n_epochs: int
        The total number of epochs we want to train for.
    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """

    weights_0 = weights_0.reshape([-1])
    weights_1 = weights_1.reshape([-1])
    result_0 = -1.0 * np.ones(weights_0.shape[0], dtype=np.float64)
    result_1 = -1.0 * np.ones(weights_1.shape[0], dtype=np.float64)
    n_samples_0 = n_epochs * (weights_0 / weights_0.max())
    n_samples_1 = n_epochs * (weights_1 / weights_1.max())
    result_0[n_samples_0 > 0] = float(n_epochs) / n_samples_0[n_samples_0 > 0]
    result_1[n_samples_1 > 0] = float(n_epochs) / n_samples_1[n_samples_1 > 0]
    result_0[n_samples_0 == 0] = n_epochs+1
    result_1[n_samples_1 == 0] = n_epochs+1
    return result_0,result_1

@numba.njit()
def clip(val):
    """Standard clamping of a value into a fixed range (in this case -4.0 to
    4.0)
    Parameters
    ----------
    val: float
        The value to be clamped.
    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val


@numba.njit("f4(f4[:],f4[:])", fastmath=True)
def rdist(x, y):
    """Reduced Euclidean distance.
    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)
    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2

    return result


@numba.njit(fastmath=True, parallel=True)
def optimize_layout(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    verbose=False,
):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).
    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.
    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.
    head: array of shape (n_1_simplices)
        The indices of the heads of 1-simplices with non-zero membership.
    tail: array of shape (n_1_simplices)
        The indices of the tails of 1-simplices with non-zero membership.
    n_epochs: int
        The number of training epochs to use in optimization.
    n_vertices: int
        The number of vertices (0-simplices) in the dataset.
    epochs_per_samples: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.
    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.
    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = head_embedding.shape[1]
    move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    for n in range(n_epochs):
        for i in range(epochs_per_sample.shape[0]):
            if epoch_of_next_sample[i] <= n:
                j = head[i]
                k = tail[i]

                current = head_embedding[j]
                other = tail_embedding[k]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                    grad_coeff /= a * pow(dist_squared, b) + 1.0
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    grad_d = clip(grad_coeff * (current[d] - other[d]))
                    current[d] += grad_d * alpha
                    if move_other:
                        other[d] += -grad_d * alpha

                epoch_of_next_sample[i] += epochs_per_sample[i]

                n_neg_samples = int(
                    (n - epoch_of_next_negative_sample[i])
                    / epochs_per_negative_sample[i]
                )

                for p in range(n_neg_samples):
                    k = tau_rand_int(rng_state) % n_vertices

                    other = tail_embedding[k]

                    dist_squared = rdist(current, other)

                    if dist_squared > 0.0:
                        grad_coeff = 2.0 * gamma * b
                        grad_coeff /= (0.001 + dist_squared) * (
                            a * pow(dist_squared, b) + 1
                        )
                    elif j == k:
                        continue
                    else:
                        grad_coeff = 0.0

                    for d in range(dim):
                        if grad_coeff > 0.0:
                            grad_d = clip(grad_coeff * (current[d] - other[d]))
                        else:
                            grad_d = 4.0
                        current[d] += grad_d * alpha

                epoch_of_next_negative_sample[i] += (
                    n_neg_samples * epochs_per_negative_sample[i]
                )

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

    return head_embedding

@numba.njit(fastmath=True, parallel=True)
def optimize_yoke(
    head_embedding1,
    tail_embedding1,
    head_embedding2,
    tail_embedding2,
    edge,
    n_epochs,
    n_vertices,
    epochs_per_sample1,
    epochs_per_sample2,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    verbose=False,
    lam=0,
    no_repul=False
):
    dim = head_embedding1.shape[1]
    move_other = head_embedding1.shape[0] == tail_embedding1.shape[0]
    alpha = initial_alpha

    epochs_per_negative_sample1 = epochs_per_sample1 / negative_sample_rate
    epochs_per_negative_sample2 = epochs_per_sample2 / negative_sample_rate
    epoch_of_next_negative_sample1 = epochs_per_negative_sample1.copy()
    epoch_of_next_negative_sample2 = epochs_per_negative_sample2.copy()
    epoch_of_next_sample1 = epochs_per_sample1.copy()
    epoch_of_next_sample2 = epochs_per_sample2.copy()
    
    alpha = alpha*10
    for n in range(n_epochs):
        for i in range(epochs_per_sample1.shape[0]):
            if epoch_of_next_sample1[i] <= n:
                j = edge[i][0]
                k = edge[i][1]
                
                current = head_embedding1[j]
                other = tail_embedding1[k]

                current_a = head_embedding2[j]
                other_a = tail_embedding2[k]
                
                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                    grad_coeff /= a * pow(dist_squared, b) + 1.0
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    grad_d = clip(grad_coeff * (current[d] - other[d]))
                    current[d] += grad_d * alpha
                    current[d] -= 2*lam*(current[d]-current_a[d])*alpha
                    if move_other:
                        other[d] += -grad_d * alpha
                        other[d] -= 2*lam*(other[d]-other_a[d])*alpha
            
                epoch_of_next_sample1[i] += epochs_per_sample1[i]
                
                if not no_repul:
                    n_neg_samples1 = int(
                        (n - epoch_of_next_negative_sample1[i])
                        / epochs_per_negative_sample1[i]
                    )


                    for p in range(n_neg_samples1):
                        k = tau_rand_int(rng_state) % n_vertices

                        other = tail_embedding1[k]

                        dist_squared = rdist(current, other)

                        if dist_squared > 0.0:
                            grad_coeff = 2.0 * gamma * b
                            grad_coeff /= (0.001 + dist_squared) * (
                                a * pow(dist_squared, b) + 1
                            )
                            #if dist_squared>5:
                                #grad_coeff=0
                        elif j == k:
                            continue
                        else:
                            grad_coeff = 0.0

                        for d in range(dim):
                            if grad_coeff > 0.0:
                                grad_d = clip(grad_coeff * (current[d] - other[d]))
                            else:
                                #grad_d = 0.0
                                grad_d = 4.0
                            current[d] += grad_d * alpha 

                    epoch_of_next_negative_sample1[i] += (
                        n_neg_samples1 * epochs_per_negative_sample1[i]
                    )
                
                else:
                    if dist_squared > 0.0:
                        grad_coeff = 2.0 * gamma * b
                        grad_coeff /= (0.001 + dist_squared) * (
                            a * pow(dist_squared, b) + 1
                        )
                    elif j == k:
                        continue
                    else:
                        grad_coeff = 0.0

                    for d in range(dim):
                        if grad_coeff > 0.0:
                            grad_d = clip(grad_coeff * (current[d] - other[d]))
                        else:
                            grad_d = 4.0
                        current[d] += grad_d * alpha

            if epoch_of_next_sample2[i] <= n:
                j2 = edge[i][0]
                k2 = edge[i][1]

                current2 = head_embedding2[j2]
                other2 = tail_embedding2[k2]

                current_b = head_embedding1[j2]
                other_b = tail_embedding1[k2]

                dist_squared = rdist(current2, other2)

                if dist_squared > 0.0:
                    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                    grad_coeff /= a * pow(dist_squared, b) + 1.0
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    grad_d = clip(grad_coeff * (current2[d] - other2[d]))
                    current2[d] += grad_d * alpha
                    current2[d] -= 2*lam*(current2[d]-current_b[d])* alpha
                    if move_other:
                        other2[d] += -grad_d * alpha
                        other2[d] -= 2*lam*(other2[d]-other_b[d])* alpha

                epoch_of_next_sample2[i] += epochs_per_sample2[i]
                
                if not no_repul:                
                    n_neg_samples2 = int(
                        (n - epoch_of_next_negative_sample2[i])/epochs_per_negative_sample2[i])

                    for p in range(n_neg_samples2):
                        k = tau_rand_int(rng_state) % n_vertices
                        other2 = tail_embedding2[k]

                        dist_squared = rdist(current2, other2)

                        if dist_squared > 0.0:
                            grad_coeff = 2.0 * gamma * b
                            grad_coeff /= (0.001 + dist_squared) * (
                                a * pow(dist_squared, b) + 1
                            )
                            #if dist_squared>5:
                                #grad_coeff=0
                        elif j == k:
                            continue
                        else:
                            grad_coeff = 0.0

                        for d in range(dim):
                            if grad_coeff > 0.0:
                                grad_d = clip(grad_coeff * (current2[d] - other2[d]))
                            else:
                                #grad_d = 0.0
                                grad_d = 4.0
                            current2[d] += grad_d * alpha 

                    epoch_of_next_negative_sample2[i] += (
                            n_neg_samples2 * epochs_per_negative_sample2[i]
                        )
                
                else:
                    if dist_squared > 0.0:
                        grad_coeff = 2.0 * gamma * b
                        grad_coeff /= (0.001 + dist_squared) * (
                            a * pow(dist_squared, b) + 1
                        )
                    elif j == k:
                        continue
                    else:
                        grad_coeff = 0.0

                    for d in range(dim):
                        if grad_coeff > 0.0:
                            grad_d = clip(grad_coeff * (current2[d] - other2[d]))
                        else:
                            grad_d = 4.0
                        current2[d] += grad_d * alpha

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")
    

    return head_embedding1,head_embedding2


@numba.njit(fastmath=True, parallel=True)
def optimize_yoke_fixed(
    head_embedding1,
    tail_embedding1,
    head_embedding2,
    tail_embedding2,
    edge,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    verbose=False,
    lam=0,
    no_repul=False
):
    dim = head_embedding1.shape[1]
    move_other = head_embedding1.shape[0] == tail_embedding1.shape[0]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()
    
    for n in range(n_epochs):
        for i in range(epochs_per_sample.shape[0]):
            if epoch_of_next_sample[i] <= n:
                j = edge[i][0]
                k = edge[i][1]
                
                current = head_embedding1[j]
                other = tail_embedding1[k]

                current_a = head_embedding2[j]
                other_a = tail_embedding2[k]
                
                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                    grad_coeff /= a * pow(dist_squared, b) + 1.0
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    grad_d = clip(grad_coeff * (current[d] - other[d]))
                    current[d] += grad_d * alpha
                    current[d] -= 2*lam*(current[d]-current_a[d])*alpha
                    if move_other:
                        other[d] += -grad_d * alpha
                        other[d] -= 2*lam*(other[d]-other_a[d])*alpha
            
                if not no_repul:
                    epoch_of_next_sample[i] += epochs_per_sample[i]

                    n_neg_samples = int(
                        (n - epoch_of_next_negative_sample[i])
                        / epochs_per_negative_sample[i]
                    )


                    for p in range(n_neg_samples):
                        k = tau_rand_int(rng_state) % n_vertices

                        other = tail_embedding1[k]

                        dist_squared = rdist(current, other)

                        if dist_squared > 0.0:
                            grad_coeff = 2.0 * gamma * b
                            grad_coeff /= (0.001 + dist_squared) * (
                                a * pow(dist_squared, b) + 1
                            )
                        elif j == k:
                            continue
                        else:
                            grad_coeff = 0.0

                        for d in range(dim):
                            if grad_coeff > 0.0:
                                grad_d = clip(grad_coeff * (current[d] - other[d]))
                            else:
                                grad_d = 4.0
                            current[d] += grad_d * alpha

                    epoch_of_next_negative_sample[i] += (
                        n_neg_samples * epochs_per_negative_sample[i]
                    )
                
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))
        
        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")
    
        #print(n,'is done')
    return head_embedding1


def simplicial_set_embedding(
    data,
    graph,
    n_components,
    initial_alpha,
    a,
    b,
    gamma,
    negative_sample_rate,
    n_epochs,
    init,
    random_state,
    metric,
    metric_kwds,
    verbose,
):
    """Perform a fuzzy simplicial set embedding, using a specified
    initialisation method and then minimizing the fuzzy set cross entropy
    between the 1-skeletons of the high and low dimensional fuzzy simplicial
    sets.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data to be embedded by UMAP.
    graph: sparse matrix
        The 1-skeleton of the high dimensional fuzzy simplicial set as
        represented by a graph for which we require a sparse matrix for the
        (weighted) adjacency matrix.
    n_components: int
        The dimensionality of the euclidean space into which to embed the data.
    initial_alpha: float
        Initial learning rate for the SGD.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    gamma: float
        Weight to apply to negative samples.
    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.
    n_epochs: int (optional, default 0)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If 0 is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    init: string
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.
    metric: string
        The metric used to measure distance in high dimensional space; used if
        multiple connected components need to be layed out.
    metric_kwds: dict
        Key word arguments to be passed to the metric function; used if
        multiple connected components need to be layed out.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized of ``graph`` into an ``n_components`` dimensional
        euclidean space.
    """
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]

    if n_epochs <= 0:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200

    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()

    if isinstance(init, str) and init == "random":
        embedding = random_state.uniform(
            low=-10.0, high=10.0, size=(graph.shape[0], n_components)
        ).astype(np.float32)
    elif isinstance(init, str) and init == "spectral":
        # We add a little noise to avoid local minima for optimization to come
        initialisation = spectral_layout(
            data,
            graph,
            n_components,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )
        expansion = 10.0 / np.abs(initialisation).max()
        embedding = (initialisation * expansion).astype(
            np.float32
        ) + random_state.normal(
            scale=0.0001, size=[graph.shape[0], n_components]
        ).astype(
            np.float32
        )
    else:
        init_data = np.array(init)
        if len(init_data.shape) == 2:
            if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
                tree = KDTree(init_data)
                dist, ind = tree.query(init_data, k=2)
                nndist = np.mean(dist[:, 1])
                embedding = init_data + np.random.normal(
                    scale=0.001 * nndist, size=init_data.shape
                ).astype(np.float32)
            else:
                embedding = init_data

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

    head = graph.row
    tail = graph.col

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    embedding = optimize_layout(
        embedding,
        embedding,
        head,
        tail,
        n_epochs,
        n_vertices,
        epochs_per_sample,
        a,
        b,
        rng_state,
        gamma,
        initial_alpha,
        negative_sample_rate,
        verbose=verbose,
    )

    return embedding

def yoke_set_embedding(
    data1,
    graph1,
    data2,
    graph2,
    n_components,
    initial_alpha,
    a,
    b,
    gamma,
    negative_sample_rate,
    n_epochs,
    init_1,
    init_2,
    random_state,
    metric,
    metric_kwds,
    verbose,
    lam,
    fixed,
    no_repul,
):

    graph1 = graph1.tocoo()
    graph1.sum_duplicates()
    if not fixed:
        graph2 = graph2.tocoo()
        graph2.sum_duplicates()
    n_vertices = graph1.shape[1]
    

    if n_epochs <= 0 and not fixed:
        # For smaller datasets we can use more epochs
        if graph1.shape[0] <= 10000 and graph2.shape[0] <=10000:
            n_epochs = 500
        else:
            n_epochs = 200
    
    elif n_epochs <= 0 and fixed:
        # For smaller datasets we can use more epochs
        if graph1.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200

    graph1.data[graph1.data < (graph1.data.max() / float(n_epochs))] = 0.0
    graph1.eliminate_zeros()
    
    graph2.data[graph2.data < (graph2.data.max() / float(n_epochs))] = 0.0
    graph2.eliminate_zeros()

    if isinstance(init_1, str) and init_1 == "random":
        embedding1 = random_state.uniform(
            low=-10.0, high=10.0, size=(graph1.shape[0], n_components)
        ).astype(np.float32)
    elif isinstance(init_1, str) and init_1 == "spectral":
        # We add a little noise to avoid local minima for optimization to come
        initialisation1 = spectral_layout(
            data1,
            graph1,
            n_components,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )
        expansion1 = 10.0 / np.abs(initialisation1).max()
        
        embedding1 = (initialisation1 * expansion1).astype(
            np.float32
        ) + random_state.normal(
            scale=0.0001, size=[graph1.shape[0], n_components]
        ).astype(
            np.float32
        )
        
    else:
        init_data_1 = np.array(init_1)
        if len(init_data_1.shape) == 2:
            if np.unique(init_data_1, axis=0).shape[0] < init_data_1.shape[0]:
                tree = KDTree(init_data_1)
                dist, ind = tree.query(init_data_1, k=2)
                nndist = np.mean(dist[:, 1])
                
                embedding1 = init_data_1 + np.random.normal(
                    scale=0.001 * nndist, size=init_data_1.shape
                ).astype(np.float32)
                
            else:
                embedding1 = init_data_1
                
                
    if not fixed:            
        if isinstance(init_2, str) and init_2 == "random":
            embedding2 = random_state.uniform(
                low=-10.0, high=10.0, size=(graph2.shape[0], n_components)
            ).astype(np.float32)
        elif isinstance(init_2, str) and init_2 == "spectral":
            # We add a little noise to avoid local minima for optimization to come
            initialisation2 = spectral_layout(
                data2,
                graph2,
                n_components,
                random_state,
                metric=metric,
                metric_kwds=metric_kwds,
            )
            expansion2 = 10.0 / np.abs(initialisation2).max()

            embedding2 = (initialisation2 * expansion2).astype(
                np.float32
            ) + random_state.normal(
                scale=0.0001, size=[graph2.shape[0], n_components]
            ).astype(
                np.float32
            )

        else:
            init_data_2 = np.array(init_2)
            if len(init_data_2.shape) == 2:
                if np.unique(init_data_2, axis=0).shape[0] < init_data_2.shape[0]:
                    tree = KDTree(init_data_2)
                    dist, ind = tree.query(init_data_2, k=2)
                    nndist = np.mean(dist[:, 1])

                    embedding2 = init_data_2 + np.random.normal(
                        scale=0.001 * nndist, size=init_data_2.shape
                    ).astype(np.float32)

                else:
                    embedding2 = init_data_2
                
                
    #epochs_per_sample1 = make_epochs_per_sample(graph1.data, n_epochs)
    #epochs_per_sample2 = make_epochs_per_sample(graph2.data, n_epochs)
    
    

    
    head1 = graph1.row
    tail1 = graph1.col
    edge1 = np.stack((head1,tail1),axis=1)
    
    if not fixed:
        head2 = graph2.row
        tail2 = graph2.col
        edge2 = np.stack((head2,tail2),axis=1)
    
    merge_ht_dir = {}
    
    if not fixed:
        for index in range(0,edge1.shape[0]):
            merge_ht_dir[tuple(edge1[index])] = [graph1.data[index],0]

        for index2 in range(0,edge2.shape[0]):
            if tuple(edge2[index2]) not in merge_ht_dir:
                merge_ht_dir[tuple(edge2[index2])] = [0,graph2.data[index2]]
            else:
                merge_ht_dir[tuple(edge2[index2])] = [merge_ht_dir[tuple(edge2[index2])][0] ,graph2.data[index2]]

        edge_0 = [[key, val[0]] for key, val in merge_ht_dir.items()]
        edge_1 = [[key, val[1]] for key, val in merge_ht_dir.items()]
        edge = np.array([item[0] for item in edge_0])
        weights_0 = np.array([[val] for _,val in edge_0])
        weights_1 = np.array([[val] for _,val in edge_1])

        epochs_per_sample1,epochs_per_sample2 = merge_epochs_per_sampler(weights_0,weights_1,n_epochs)


        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        #pdb.set_trace()
        embedding1,embedding2 = optimize_yoke(
            embedding1,
            embedding1,
            embedding2,
            embedding2,
            edge,
            n_epochs,
            n_vertices,
            epochs_per_sample1,
            epochs_per_sample2,
            a,
            b,
            rng_state,
            gamma,
            initial_alpha,
            negative_sample_rate,
            verbose=verbose,
            lam=lam,
            no_repul=no_repul,
        )
    
    else:
        epochs_per_sample = make_epochs_per_sample(graph1.data, n_epochs)

        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
        embedding = optimize_yoke_fixed(
            embedding1,
            embedding1,
            data2,
            data2,
            edge1,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            initial_alpha,
            negative_sample_rate,
            verbose=verbose,
            lam = lam,
            no_repul=no_repul,
    )
        embedding1 = embedding
        embedding2 = data2

    return embedding1,embedding2


@numba.njit()
def init_transform(indices, weights, embedding):
    """Given indices and weights and an original embeddings
    initialize the positions of new points relative to the
    indices and weights (of their neighbors in the source data).
    Parameters
    ----------
    indices: array of shape (n_new_samples, n_neighbors)
        The indices of the neighbors of each new sample
    weights: array of shape (n_new_samples, n_neighbors)
        The membership strengths of associated 1-simplices
        for each of the new samples.
    embedding: array of shape (n_samples, dim)
        The original embedding of the source data.
    Returns
    -------
    new_embedding: array of shape (n_new_samples, dim)
        An initial embedding of the new sample points.
    """
    result = np.zeros((indices.shape[0], embedding.shape[1]), dtype=np.float32)

    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            for d in range(embedding.shape[1]):
                result[i, d] += weights[i, j] * embedding[indices[i, j], d]

    return result


def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


class UMAP(BaseEstimator):
    """Uniform Manifold Approximation and Projection
    Finds a low dimensional embedding of the data that approximates
    an underlying manifold.
    Parameters
    ----------
    n_neighbors: float (optional, default 15)
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.
    n_components: int (optional, default 2)
        The dimension of the space to embed into. This defaults to 2 to
        provide easy visualization, but can reasonably be set to any
        integer value in the range 2 to 100.
    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.
    n_epochs: int (optional, default None)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    learning_rate: float (optional, default 1.0)
        The initial learning rate for the embedding optimization.
    init: string (optional, default 'spectral')
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.
    min_dist: float (optional, default 0.1)
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.
    spread: float (optional, default 1.0)
        The effective scale of embedded points. In combination with ``min_dist``
        this determines how clustered/clumped the embedded points are.
    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.
    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
    repulsion_strength: float (optional, default 1.0)
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.
    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.
    transform_queue_size: float (optional, default 4.0)
        For transform operations (embedding new points using a trained model_
        this will control how aggressively to search for nearest neighbors.
        Larger values will result in slower performance but more accurate
        nearest neighbor evaluation.
    a: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    b: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    random_state: int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    metric_kwds: dict (optional, default None)
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance. If None then no arguments are passed on.
    angular_rp_forest: bool (optional, default False)
        Whether to use an angular random projection forest to initialise
        the approximate nearest neighbor search. This can be faster, but is
        mostly on useful for metric that use an angular style distance such
        as cosine, correlation etc. In the case of those metrics angular forests
        will be chosen automatically.
    target_n_neighbors: int (optional, default -1)
        The number of nearest neighbors to use to construct the target simplcial
        set. If set to -1 use the ``n_neighbors`` value.
    target_metric: string or callable (optional, default 'categorical')
        The metric used to measure distance for a target array is using supervised
        dimension reduction. By default this is 'categorical' which will measure
        distance in terms of whether categories match or are different. Furthermore,
        if semi-supervised is required target values of -1 will be trated as
        unlabelled under the 'categorical' metric. If the target array takes
        continuous values (e.g. for a regression problem) then metric of 'l1'
        or 'l2' is probably more appropriate.
    target_metric_kwds: dict (optional, default None)
        Keyword argument to pass to the target metric when performing
        supervised dimension reduction. If None then no arguments are passed on.
    target_weight: float (optional, default 0.5)
        weighting factor between data topology and target topology. A value of
        0.0 weights entirely on data, a value of 1.0 weights entirely on target.
        The default of 0.5 balances the weighting equally between data and target.
    transform_seed: int (optional, default 42)
        Random seed used for the stochastic aspects of the transform operation.
        This ensures consistency in transform operations.
    verbose: bool (optional, default False)
        Controls verbosity of logging.
    """

    def __init__(
        self,
        n_neighbors=15,
        n_components=2,
        metric="euclidean",
        n_epochs=None,
        learning_rate=1.0,
        init="spectral",
        init_1="spectral",
        init_2="spectral",
        min_dist=0.1,
        spread=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        transform_queue_size=4.0,
        a=None,
        b=None,
        random_state=None,
        metric_kwds=None,
        angular_rp_forest=False,
        target_n_neighbors=-1,
        target_metric="categorical",
        target_metric_kwds=None,
        target_weight=0.5,
        transform_seed=42,
        verbose=False,
        lam = 0
    ):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.n_epochs = n_epochs
        self.init = init
        self.init_1 = init_1
        self.init_2 = init_2
        self.n_components = n_components
        self.repulsion_strength = repulsion_strength
        self.learning_rate = learning_rate

        self.spread = spread
        self.min_dist = min_dist
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.transform_queue_size = transform_queue_size
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_metric_kwds = target_metric_kwds
        self.target_weight = target_weight
        self.transform_seed = transform_seed
        self.verbose = verbose

        self.a = a
        self.b = b
        
        self.lam = lam
        
    def _validate_parameters(self):
        if self.set_op_mix_ratio < 0.0 or self.set_op_mix_ratio > 1.0:
            raise ValueError("set_op_mix_ratio must be between 0.0 and 1.0")
        if self.repulsion_strength < 0.0:
            raise ValueError("repulsion_strength cannot be negative")
        if self.min_dist > self.spread:
            raise ValueError("min_dist must be less than or equal to spread")
        if self.min_dist < 0.0:
            raise ValueError("min_dist must be greater than 0.0")
        if not isinstance(self.init, str) and not isinstance(self.init, np.ndarray):
            raise ValueError("init must be a string or ndarray")
        if isinstance(self.init, str) and self.init not in ("spectral", "random"):
            raise ValueError('string init values must be "spectral" or "random"')
        if (
            isinstance(self.init, np.ndarray)
            and self.init.shape[1] != self.n_components
        ):
            raise ValueError("init ndarray must match n_components value")
        if not isinstance(self.metric, str) and not callable(self.metric):
            raise ValueError("metric must be string or callable")
        if self.negative_sample_rate < 0:
            raise ValueError("negative sample rate must be positive")
        if self._initial_alpha < 0.0:
            raise ValueError("learning_rate must be positive")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 2")
        if self.target_n_neighbors < 2 and self.target_n_neighbors != -1:
            raise ValueError("target_n_neighbors must be greater than 2")
        if not isinstance(self.n_components, int):
            raise ValueError("n_components must be an int")
        if self.n_components < 1:
            raise ValueError("n_components must be greater than 0")
        if self.n_epochs is not None and (
            self.n_epochs <= 10 or not isinstance(self.n_epochs, int)
        ):
            raise ValueError("n_epochs must be a positive integer "
                             "larger than 10")

    def fit(self, X, y=None):
        """Fit X into an embedded space.
        Optionally use y for supervised dimension reduction.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.
        y : array, shape (n_samples)
            A target array for supervised dimension reduction. How this is
            handled is determined by parameters UMAP was instantiated with.
            The relevant attributes are ``target_metric`` and
            ``target_metric_kwds``.
        """

        X = check_array(X, dtype=np.float32, accept_sparse="csr")
        self._raw_data = X

        # Handle all the optional arguments, setting default
        if self.a is None or self.b is None:
            self._a, self._b = find_ab_params(self.spread, self.min_dist)
        else:
            self._a = self.a
            self._b = self.b

        if self.metric_kwds is not None:
            self._metric_kwds = self.metric_kwds
        else:
            self._metric_kwds = {}

        if self.target_metric_kwds is not None:
            self._target_metric_kwds = self.target_metric_kwds
        else:
            self._target_metric_kwds = {}

        if isinstance(self.init, np.ndarray):
            init = check_array(self.init, dtype=np.float32, accept_sparse=False)
        else:
            init = self.init

        self._initial_alpha = self.learning_rate

        self._validate_parameters()

        if self.verbose:
            print(str(self))

        # Error check n_neighbors based on data size
        if X.shape[0] <= self.n_neighbors:
            if X.shape[0] == 1:
                self.embedding_ = np.zeros((1, self.n_components))  # needed to sklearn comparability
                return self

            warn(
                "n_neighbors is larger than the dataset size; truncating to "
                "X.shape[0] - 1"
            )
            self._n_neighbors = X.shape[0] - 1
        else:
            self._n_neighbors = self.n_neighbors

        if scipy.sparse.isspmatrix_csr(X):
            if not X.has_sorted_indices:
                X.sort_indices()
            self._sparse_data = True
        else:
            self._sparse_data = False

        random_state = check_random_state(self.random_state)

        if self.verbose:
            print("Construct fuzzy simplicial set")

        # Handle small cases efficiently by computing all distances
        if X.shape[0] < 4096:
            self._small_data = True
            dmat = pairwise_distances(X, metric=self.metric, **self._metric_kwds)
            self.graph_ = fuzzy_simplicial_set(
                dmat,
                self._n_neighbors,
                random_state,
                "precomputed",
                self._metric_kwds,
                None,
                None,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                self.verbose,
            )
        else:
            self._small_data = False
            # Standard case
            (self._knn_indices, self._knn_dists, self._rp_forest) = nearest_neighbors(
                X,
                self._n_neighbors,
                self.metric,
                self._metric_kwds,
                self.angular_rp_forest,
                random_state,
                self.verbose,
            )

            self.graph_ = fuzzy_simplicial_set(
                X,
                self.n_neighbors,
                random_state,
                self.metric,
                self._metric_kwds,
                self._knn_indices,
                self._knn_dists,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                self.verbose,
            )

            self._search_graph = scipy.sparse.lil_matrix(
                (X.shape[0], X.shape[0]), dtype=np.int8
            )
            self._search_graph.rows = self._knn_indices
            self._search_graph.data = (self._knn_dists != 0).astype(np.int8)
            self._search_graph = self._search_graph.maximum(
                self._search_graph.transpose()
            ).tocsr()

            if callable(self.metric):
                self._distance_func = self.metric
            elif self.metric in dist.named_distances:
                self._distance_func = dist.named_distances[self.metric]
            elif self.metric == 'precomputed':
                warn('Using precomputed metric; transform will be unavailable for new data')
            else:
                raise ValueError(
                    "Metric is neither callable, " + "nor a recognised string"
                )

            if self.metric != 'precomputed':
                self._dist_args = tuple(self._metric_kwds.values())

                self._random_init, self._tree_init = make_initialisations(
                    self._distance_func, self._dist_args
                )
                self._search = make_initialized_nnd_search(
                    self._distance_func, self._dist_args
                )

        if y is not None:
            if len(X) != len(y):
                raise ValueError(
                    "Length of x = {len_x}, length of y = {len_y}, while it must be equal.".format(
                        len_x=len(X), len_y=len(y)
                    )
                )
            y_ = check_array(y, ensure_2d=False)
            if self.target_metric == "categorical":
                if self.target_weight < 1.0:
                    far_dist = 2.5 * (1.0 / (1.0 - self.target_weight))
                else:
                    far_dist = 1.0e12
                self.graph_ = categorical_simplicial_set_intersection(
                    self.graph_, y_, far_dist=far_dist
                )
            else:
                if self.target_n_neighbors == -1:
                    target_n_neighbors = self._n_neighbors
                else:
                    target_n_neighbors = self.target_n_neighbors

                # Handle the small case as precomputed as before
                if y.shape[0] < 4096:
                    ydmat = pairwise_distances(y_[np.newaxis, :].T,
                                               metric=self.target_metric,
                                               **self._target_metric_kwds)
                    target_graph = fuzzy_simplicial_set(
                        ydmat,
                        target_n_neighbors,
                        random_state,
                        "precomputed",
                        self._target_metric_kwds,
                        None,
                        None,
                        False,
                        1.0,
                        1.0,
                        False
                    )
                else:
                    # Standard case
                    target_graph = fuzzy_simplicial_set(
                        y_[np.newaxis, :].T,
                        target_n_neighbors,
                        random_state,
                        self.target_metric,
                        self._target_metric_kwds,
                        None,
                        None,
                        False,
                        1.0,
                        1.0,
                        False,
                    )
                # product = self.graph_.multiply(target_graph)
                # # self.graph_ = 0.99 * product + 0.01 * (self.graph_ +
                # #                                        target_graph -
                # #                                        product)
                # self.graph_ = product
                self.graph_ = general_simplicial_set_intersection(
                    self.graph_, target_graph, self.target_weight
                )
                self.graph_ = reset_local_connectivity(self.graph_)

        if self.n_epochs is None:
            n_epochs = 0
        else:
            n_epochs = self.n_epochs

        if self.verbose:
            print(ts(), "Construct embedding")

        self.embedding_ = simplicial_set_embedding(
            self._raw_data,
            self.graph_,
            self.n_components,
            self._initial_alpha,
            self._a,
            self._b,
            self.repulsion_strength,
            self.negative_sample_rate,
            n_epochs,
            init,
            random_state,
            self.metric,
            self._metric_kwds,
            self.verbose,
        )

        if self.verbose:
            print(ts() + " Finished embedding")

        self._input_hash = joblib.hash(self._raw_data)

        return self

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed
        output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.
        y : array, shape (n_samples)
            A target array for supervised dimension reduction. How this is
            handled is determined by parameters UMAP was instantiated with.
            The relevant attributes are ``target_metric`` and
            ``target_metric_kwds``.
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        self.fit(X, y)
        return self.embedding_
    
    def yoke(self, X1,X2, y=None,fixed=False,no_repul=False):
        
        X1 = check_array(X1, dtype=np.float32, accept_sparse="csr")
        X2 = check_array(X2, dtype=np.float32, accept_sparse="csr")
        self._raw_data1 = X1
        self._raw_data2 = X2

        # Handle all the optional arguments, setting default
        if self.a is None or self.b is None:
            self._a, self._b = find_ab_params(self.spread, self.min_dist)
        else:
            self._a = self.a
            self._b = self.b

        if self.metric_kwds is not None:
            self._metric_kwds = self.metric_kwds
        else:
            self._metric_kwds = {}

        if self.target_metric_kwds is not None:
            self._target_metric_kwds = self.target_metric_kwds
        else:
            self._target_metric_kwds = {}

        if isinstance(self.init_1, np.ndarray):
            init_1 = check_array(self.init_1, dtype=np.float32, accept_sparse=False)
        else:
            init_1 = self.init_1
        
        if isinstance(self.init_2, np.ndarray):
            init_2 = check_array(self.init_2, dtype=np.float32, accept_sparse=False)
        else:
            init_2 = self.init_2

        self._initial_alpha = self.learning_rate

        self._validate_parameters()

        if self.verbose:
            print(str(self))

        # Error check n_neighbors based on data size
        if X1.shape[0] <= self.n_neighbors:
            if X1.shape[0] == 1:
                self.embedding_ = np.zeros((1, self.n_components))  # needed to sklearn comparability
                return self

            warn(
                "n_neighbors is larger than the dataset size; truncating to "
                "X1.shape[0] - 1"
            )
            self._n_neighbors = X1.shape[0] - 1
        else:
            self._n_neighbors = self.n_neighbors

        if scipy.sparse.isspmatrix_csr(X1):
            if not X1.has_sorted_indices:
                X1.sort_indices()
            self._sparse_data = True
        else:
            self._sparse_data = False

        random_state = check_random_state(self.random_state)

        if self.verbose:
            print("Construct fuzzy simplicial set")

        # Handle small cases efficiently by computing all distances
        if X1.shape[0] < 4096:
            self._small_data = True
            dmat1 = pairwise_distances(X1, metric=self.metric, **self._metric_kwds)
            self.graph_1 = fuzzy_simplicial_set(
                dmat1,
                self._n_neighbors,
                random_state,
                "precomputed",
                self._metric_kwds,
                None,
                None,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                self.verbose,
            )
            
            dmat2 = pairwise_distances(X2, metric=self.metric, **self._metric_kwds)
            self.graph_2 = fuzzy_simplicial_set(
                dmat2,
                self._n_neighbors,
                random_state,
                "precomputed",
                self._metric_kwds,
                None,
                None,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                self.verbose,
            )
                
        else:
            self._small_data = False
            # Standard case
            (self._knn_indices1, self._knn_dists1, self._rp_forest1) = nearest_neighbors(
                X1,
                self._n_neighbors,
                self.metric,
                self._metric_kwds,
                self.angular_rp_forest,
                random_state,
                self.verbose,
            )

            self.graph_1 = fuzzy_simplicial_set(
                X1,
                self.n_neighbors,
                random_state,
                self.metric,
                self._metric_kwds,
                self._knn_indices1,
                self._knn_dists1,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                self.verbose,
            )
            
            #self.graph_2 = None
            (self._knn_indices2, self._knn_dists2, self._rp_forest2) = nearest_neighbors(
                X2,
                self._n_neighbors,
                self.metric,
                self._metric_kwds,
                self.angular_rp_forest,
                random_state,
                self.verbose,
            )

            self.graph_2 = fuzzy_simplicial_set(
                X2,
                self.n_neighbors,
                random_state,
                self.metric,
                self._metric_kwds,
                self._knn_indices2,
                self._knn_dists2,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                self.verbose,
            )

            self._search_graph1 = scipy.sparse.lil_matrix(
                (X1.shape[0], X1.shape[0]), dtype=np.int8
            )
            self._search_graph1.rows = self._knn_indices1
            self._search_graph1.data = (self._knn_dists1 != 0).astype(np.int8)
            self._search_graph1 = self._search_graph1.maximum(
                self._search_graph1.transpose()
            ).tocsr()
            
            self._search_graph2 = scipy.sparse.lil_matrix(
                (X2.shape[0], X2.shape[0]), dtype=np.int8
            )
            self._search_graph2.rows = self._knn_indices2
            self._search_graph2.data = (self._knn_dists2!= 0).astype(np.int8)
            self._search_graph2= self._search_graph2.maximum(
                self._search_graph2.transpose()
            ).tocsr()

            if callable(self.metric):
                self._distance_func = self.metric
            elif self.metric in dist.named_distances:
                self._distance_func = dist.named_distances[self.metric]
            elif self.metric == 'precomputed':
                warn('Using precomputed metric; transform will be unavailable for new data')
            else:
                raise ValueError(
                    "Metric is neither callable, " + "nor a recognised string"
                )

            if self.metric != 'precomputed':
                self._dist_args = tuple(self._metric_kwds.values())

                self._random_init, self._tree_init = make_initialisations(
                    self._distance_func, self._dist_args
                )
                self._search = make_initialized_nnd_search(
                    self._distance_func, self._dist_args
                )

        if y is not None:
            if len(X1) != len(y):
                raise ValueError(
                    "Length of x = {len_x}, length of y = {len_y}, while it must be equal.".format(
                        len_x=len(X1), len_y=len(y)
                    )
                )
            y_ = check_array(y, ensure_2d=False)
            if self.target_metric == "categorical":
                if self.target_weight < 1.0:
                    far_dist = 2.5 * (1.0 / (1.0 - self.target_weight))
                else:
                    far_dist = 1.0e12
                self.graph_ = categorical_simplicial_set_intersection(
                    self.graph_, y_, far_dist=far_dist
                )
            else:
                if self.target_n_neighbors == -1:
                    target_n_neighbors = self._n_neighbors
                else:
                    target_n_neighbors = self.target_n_neighbors

                # Handle the small case as precomputed as before
                if y.shape[0] < 4096:
                    ydmat = pairwise_distances(y_[np.newaxis, :].T,
                                               metric=self.target_metric,
                                               **self._target_metric_kwds)
                    target_graph = fuzzy_simplicial_set(
                        ydmat,
                        target_n_neighbors,
                        random_state,
                        "precomputed",
                        self._target_metric_kwds,
                        None,
                        None,
                        False,
                        1.0,
                        1.0,
                        False
                    )
                else:
                    # Standard case
                    target_graph = fuzzy_simplicial_set(
                        y_[np.newaxis, :].T,
                        target_n_neighbors,
                        random_state,
                        self.target_metric,
                        self._target_metric_kwds,
                        None,
                        None,
                        False,
                        1.0,
                        1.0,
                        False,
                    )
                # product = self.graph_.multiply(target_graph)
                # # self.graph_ = 0.99 * product + 0.01 * (self.graph_ +
                # #                                        target_graph -
                # #                                        product)
                # self.graph_ = product
                self.graph_ = general_simplicial_set_intersection(
                    self.graph_, target_graph, self.target_weight
                )
                self.graph_ = reset_local_connectivity(self.graph_)

        if self.n_epochs is None:
            n_epochs = 0
        else:
            n_epochs = self.n_epochs

        if self.verbose:
            print(ts(), "Construct embedding")
        
        
        self.embedding_1,self.embedding_2 = yoke_set_embedding(self._raw_data1,
            self.graph_1,self._raw_data2,self.graph_2,
            self.n_components,
            self._initial_alpha,
            self._a,
            self._b,
            self.repulsion_strength,
            self.negative_sample_rate,
            n_epochs,
            init_1,
            init_2,                                            
            random_state,
            self.metric,
            self._metric_kwds,
            self.verbose,self.lam,fixed,no_repul,)

        if self.verbose:
            print(ts() + " Finished embedding")

        #self._input_hash = joblib.hash(self._raw_data)

        return self.embedding_1,self.embedding_2

    
    def get_yoke_loss(self):
        if self.graph_2 is None:
            return cal_loss(self.embedding_1,self.graph_1,self._a,self._b),None
        return cal_loss(self.embedding_1,self.graph_1,self._a,self._b),cal_loss(self.embedding_2,self.graph_2,self._a,self._b)
    
    def get_loss(self):
        return cal_loss(self.embedding_,self.graph_,self._a,self._b)
    
    def get_semi_loss(self):
        return compute_yoke_loss_semi(self.embedding_1,self.embedding_2,self.graph_1.toarray(),self.graph_2.toarray(),self._a,self._b)
    
    def yoke_transform(self,X1,X2,y=None,fixed=False,no_repul=False):
        embedding1,embedding2 = self.yoke(X1,X2,y,fixed,no_repul)
        return embedding1,embedding2

    def transform(self, X):
        """Transform X into the existing embedded space and return that
        transformed output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            New data to be transformed.
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the new data in low-dimensional space.
        """
        # If we fit just a single instance then error
        if self.embedding_.shape[0] == 1:
            raise ValueError('Transform unavailable when model was fit with'
                             'only a single data sample.')
        # If we just have the original input then short circuit things
        X = check_array(X, dtype=np.float32, accept_sparse="csr")
        x_hash = joblib.hash(X)
        if x_hash == self._input_hash:
            return self.embedding_

        if self._sparse_data:
            raise ValueError("Transform not available for sparse input.")
        elif self.metric == 'precomputed':
            raise ValueError("Transform  of new data not available for "
                             "precomputed metric.")

        X = check_array(X, dtype=np.float32, order="C")
        random_state = check_random_state(self.transform_seed)
        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        if self._small_data:
            dmat = pairwise_distances(
                X, self._raw_data, metric=self.metric, **self._metric_kwds
            )
            indices = np.argpartition(dmat,
                                      self._n_neighbors)[:, :self._n_neighbors]
            dmat_shortened = submatrix(dmat, indices, self._n_neighbors)
            indices_sorted = np.argsort(dmat_shortened)
            indices = submatrix(indices, indices_sorted, self._n_neighbors)
            dists = submatrix(dmat_shortened, indices_sorted,
                              self._n_neighbors)
        else:
            init = initialise_search(
                self._rp_forest,
                self._raw_data,
                X,
                int(self._n_neighbors * self.transform_queue_size),
                self._random_init,
                self._tree_init,
                rng_state,
            )
            result = self._search(
                self._raw_data,
                self._search_graph.indptr,
                self._search_graph.indices,
                init,
                X,
            )

            indices, dists = deheap_sort(result)
            indices = indices[:, : self._n_neighbors]
            dists = dists[:, : self._n_neighbors]

        adjusted_local_connectivity = max(0, self.local_connectivity - 1.0)
        sigmas, rhos = smooth_knn_dist(
            dists, self._n_neighbors, local_connectivity=adjusted_local_connectivity
        )

        rows, cols, vals = compute_membership_strengths(indices, dists, sigmas, rhos)

        graph = scipy.sparse.coo_matrix(
            (vals, (rows, cols)), shape=(X.shape[0], self._raw_data.shape[0])
        )

        # This was a very specially constructed graph with constant degree.
        # That lets us do fancy unpacking by reshaping the csr matrix indices
        # and data. Doing so relies on the constant degree assumption!
        csr_graph = normalize(graph.tocsr(), norm="l1")
        inds = csr_graph.indices.reshape(X.shape[0], self._n_neighbors)
        weights = csr_graph.data.reshape(X.shape[0], self._n_neighbors)
        embedding = init_transform(inds, weights, self.embedding_)

        if self.n_epochs is None:
            # For smaller datasets we can use more epochs
            if graph.shape[0] <= 10000:
                n_epochs = 100
            else:
                n_epochs = 30
        else:
            n_epochs = self.n_epochs // 3.0

        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
        graph.eliminate_zeros()

        epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

        head = graph.row
        tail = graph.col

        embedding = optimize_layout(
            embedding,
            self.embedding_,
            head,
            tail,
            n_epochs,
            graph.shape[1],
            epochs_per_sample,
            self._a,
            self._b,
            rng_state,
            self.repulsion_strength,
            self._initial_alpha,
            self.negative_sample_rate,
            verbose=self.verbose,
        )

        return embedding

######################################################


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
        self.learning_rate = 0.0045
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
        print(x0.shape)
        x1= self.model2(x1)
        x1 = F.avg_pool2d(x1, 8)
        x1=x1.view(bs,-1)
        print(x1.shape)
        x=torch.cat([x0,x1],1)
        if self.training==True:
            x=F.dropout(x,0.4)
        print(x.shape)
        print("here")
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
        # data_dir = '/lab/vislab/OPEN/iNaturalist/train_val2019/Plants/'

        train_path = '/lab/vislab/OPEN/datasets_RGB_new/train/'
        valid_path = '/lab/vislab/OPEN/datasets_RGB_new/val/'

        self.trainset = torchvision.datasets.ImageFolder(train_path, data_transforms['train'])
        self.validset = torchvision.datasets.ImageFolder(valid_path, data_transforms['valid'])


        # master_dataset = datasets.ImageFolder(
        #     data_dir, data_transforms['train'],
        # )

        # self.trainset, self.validset = torch.utils.data.random_split(master_dataset, (126770,31693))

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=25, shuffle=False, num_workers=25)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=25, shuffle=False, num_workers=25)
    
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
        # neptune.log_metric('train_loss', train_loss)
        # neptune.log_metric('train acc', train_acc)

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

        # neptune.log_metric('val_loss', val_loss)
        # neptune.log_metric('val acc', val_acc)


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
        return random.sample(crs,4)+[re(Rand(img))]

    def RandomErase(self, img, p=0.5, s=(0.06,0.12), r=(0.5,1.5)):
        im=np.array(img)
        w,h,_=im.shape
        S=w*h
        pi=random()
        if pi>p:
            return img
        else:
            Se=S*(random.random()*(s[1]-s[0])+s[0])
            re=random()*(r[1]-r[0])+r[0]
            He=int(np.sqrt(Se*re))
            We=int(np.sqrt(Se/re))
            if He>=h:
                He=h-1
            if We>=w:
                We=w-1
            xe=int(random.random()*(w-We))
            ye=int(random.random()*(h-He))
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
    # logger = pl_loggers.TensorBoardLogger('/lab/vislab/OPEN/justin/lightning-OPEN/iNat_logs/v2/')
    metrics_callback = MetricCallback()
    # logger = pl_loggers.TensorBoardLogger('/lab/vislab/OPEN/justin/lightning-OPEN/logs/layer_4/')
    trainer = pl.Trainer(
        max_epochs=15,
        num_sanity_val_steps=-1,
        gpus=[2] if torch.cuda.is_available() else None,
        callbacks=[metrics_callback],
        # logger=logger
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