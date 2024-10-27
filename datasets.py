import numpy as np
import pickle as pk
import csv

import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

def readDataset(dataset):
    # read npy files from folder dataMats
    x_path='./dataMats/'+dataset+'_X.npy'
    y_path='./dataMats/'+dataset+'_Y.npy'        
    X=np.load(x_path)
    Y=np.load(y_path)

    return X,Y

class myDataset(Dataset):
    def __init__(self, x,y, transform=None, target_transform=None):
        self.x=x.type(torch.float)
        self.y=y.type(torch.float)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        instance=self.x[idx,:]
        label=self.y[idx]
        if self.transform:
            instance = self.transform(instance)
        if self.target_transform:
            label = self.target_transform(label)
        return instance, label

def split(data,label=None,train_rate=0.1,candidate_rate=0.6,\
          test_rate=0.3,seed=25,even=False):
    # initial split for AL
    # both train and test include all labels (if possible) in this version
    index=np.arange(data.shape[0])
    np.random.seed(seed)
    np.random.shuffle(index)
    if(even is False):#do not require each label have a positive instance in train.
        train_index=index[0:int(len(index)*train_rate)]
        candidate_index=index[int(len(index)*train_rate):int(len(index)*(train_rate+candidate_rate))]
        test_index=index[int(len(index)*(train_rate+candidate_rate)):]
        return list(train_index),list(candidate_index),list(test_index)
    elif(even is True):#scan index to determine the train index first. 
        label_stack=label[0,:]#label_stack is 0/1 L length vector, indicating whether a label has already appreared in the training
        train_index=[index[0]]
        orilen = len(index)
        i=0
        while (len(train_index)<int(len(index)*train_rate)):
            i=i+1
            current_label=np.sum(label_stack)
            if(current_label<label.shape[1]):  #then need a new training data with new positive label          
                updated_label=np.sum(np.logical_or(label[index[i],:],label_stack))
                if(updated_label>current_label):#if introducing the next data introduce a new label(s),add it. 
                    train_index.append(index[i])
                    label_stack=np.logical_or(label[index[i],:],label_stack)#update label stack
                    #print(np.sum(label_stack))
                else:#skip this data point
                    pass
            else:
                train_index.append(index[i])
        #delete the train index from index
        index=[x for x in index if x not in train_index]
        test_index = [index[0]]
        label_stack=label[test_index[0],:]
        i=0
        while (len(test_index)<int(orilen*test_rate)):
            i=i+1
            current_label=np.sum(label_stack)
            if(current_label<label.shape[1]):  #then need a new training data with new positive label          
                updated_label=np.sum(np.logical_or(label[index[i],:],label_stack))
                if(updated_label>current_label):#if introducing the next data introduce a new label(s),add it. 
                    test_index.append(index[i])
                    label_stack=np.logical_or(label[index[i],:],label_stack)#update label stack
                    #print(np.sum(label_stack))
                else:#skip this data point
                    pass
            else:
                test_index.append(index[i])

        #delete the candidate index from index
        candidate_index=[x for x in index if x not in test_index]
        candidate_index = candidate_index[:int(orilen*candidate_rate)]
        return list(train_index),list(candidate_index),list(test_index)

'''
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn.gaussian_process.kernels import RBF
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
import os

wnew = 10
# from B2M_util import *

#determine k, the number of component automatically.
def deterministic(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def syntheticDataGen(xLeft,xRight,yUp,yDown,nPerRow,sigma2=1.5,samplePerClass=100,thres=0.25,dim=2,additionalRule=False):
    #centorids of the Gaussian clusters are arranged in the grid range(xLeft,xRight,nPerRow),range(yLeft,yRight,nPerRow)
    samples=[]
    pdfs=[]
    clusterIndex=[]#used for manually labeling a specific cluster
    nClusters=0
    restDims=[0 for i in range(dim-2)]#all centers of Gaussian in the same 2D plane. So the restDims are dim-2
    for yInd,centerY in enumerate(np.arange(yUp,yDown,-nPerRow)):
        for xInd,centerX in enumerate(np.arange(xLeft,xRight,nPerRow)):
            nClusters+=1
            a=np.random.multivariate_normal([centerX,centerY]+restDims, np.eye(dim)*sigma2, samplePerClass)
            n=multivariate_normal(mean=[centerX,centerY]+restDims, cov = np.eye(dim)*sigma2)
            samples.append(a)
            pdfs.append(n)
            clusterIndex+=[(xInd,yInd) for i in range(samplePerClass)]
    x=np.concatenate(samples,axis=0)            
    y=np.zeros([samplePerClass*nClusters,int(0.5*nClusters)+1])
    trueProb=np.zeros([samplePerClass*nClusters,nClusters]) 
    print('nclusters:%d'%(nClusters))
    for i in range(nClusters):
        if(i<nClusters*0.5):
            y[i*samplePerClass:(i+1)*samplePerClass,i]=1    
        else:#repeat the labeling
            y[i*samplePerClass:(i+1)*samplePerClass,int(i-nClusters*0.5)]=1   
        trueProb[:,i]=pdfs[i].pdf(x)
    trueProb=normalize(trueProb,norm='l1') 
    for i in range(x.shape[0]):
        for j in range(len(pdfs)):
            if(trueProb[i,j]>thres):
                if(j<int(0.5*nClusters)):
                    y[i,j]=1
                else:
                    y[i,int(j-nClusters*0.5)]=1
    print(y.shape)
    def labelCluster(clusterList,labelList):
        return
    if(additionalRule):
        #randomly pick 10% data to generate 2 new labels
        cardi=np.sum(y,axis=1)
        #add labels indicating cardi=2 and cardi=4
        y=np.concatenate([y,np.zeros([y.shape[0],2])],axis=1)
        for i in range(y.shape[0]):
            if(cardi[i]==3):
                y[i,-2]=1
            elif(cardi[i]>3):
                y[i,-1]=1
        #add two logical labels wrt l1 l2, l2 l3,  labels(and or opertation)
        y=np.concatenate([y,np.zeros([y.shape[0],8])],axis=1)
        for i in range(y.shape[0]):
            if(y[i,0] and y[i,1]): #over lapping zone label. 
                y[i,-8]=1
            if(y[i,1] and y[i,2]): # over lapping zone label
                y[i,-7]=1
            if(y[i,2] and y[i,3]): #over lapping zone label. 
                y[i,-6]=1
            if(y[i,4] and y[i,5]): # over lapping zone label
                y[i,-5]=1
        
        tem=y[:,0]
        for i in [5]:
            tem=np.logical_or(tem,y[:,i])#index of data with label 0,5
        index=np.where(tem==1)[0]
        np.random.shuffle(index)
        #the 'contain' relationship
        y[index,-4]=1 #data with label 0 or 5 has this label -5
        y[index[0:int(0.3*len(index))],-3]=1#1/3 of the data with label -5 has label -6
        np.random.shuffle(index)
        y[index[0:int(0.3*len(index))],-2]=1 #1/3 of the data with label -5 has label -7        
        #exclusive relationship
        print('what')
        for i in range(y.shape[0]):
            if(y[i,-2] and not y[i,-3]):
                y[i,-1]=1
    print('label sparsity:%f'%(np.sum(y)/(y.shape[0]*y.shape[1])))
    print('number of instance cardi=2: %f cardi=3 %f cardi=4 %f cardi=5 %f' %(len(np.where(np.sum(y,axis=1)==2)[0]),len(np.where(np.sum(y,axis=1)==3)[0]),len(np.where(np.sum(y,axis=1)==4)[0]),len(np.where(np.sum(y,axis=1)==5)[0])))
    return x,y,pdfs,clusterIndex,trueProb

def syntheticDataGen_label(xLeft,xRight,yUp,yDown,nPerRow,sigma2=1.5,samplePerClass=100,thres=0.25,dim=2,additionalRule=False):
    #centorids of the Gaussian clusters are arranged in the grid range(xLeft,xRight,nPerRow),range(yLeft,yRight,nPerRow)
    samples=[]
    pdfs=[]
    clusterIndex=[]#used for manually labeling a specific cluster
    nClusters=0
    restDims=[0 for i in range(dim-2)]#all centers of Gaussian in the same 2D plane. So the restDims are dim-2
    for yInd,centerY in enumerate(np.arange(yUp,yDown,-nPerRow)):
        for xInd,centerX in enumerate(np.arange(xLeft,xRight,nPerRow)):
            nClusters+=1
            a=np.random.multivariate_normal([centerX,centerY]+restDims, np.eye(dim)*sigma2, samplePerClass)
            n=multivariate_normal(mean=[centerX,centerY]+restDims, cov = np.eye(dim)*sigma2)
            samples.append(a)
            pdfs.append(n)
            clusterIndex+=[(xInd,yInd) for i in range(samplePerClass)]
    x=np.concatenate(samples,axis=0)            
    y=np.zeros([samplePerClass*nClusters,int(0.5*nClusters)+1])
    trueProb=np.zeros([samplePerClass*nClusters,nClusters]) 
    print('nclusters:%d'%(nClusters))
    for i in range(nClusters):
        if(i<nClusters*0.5):
            y[i*samplePerClass:(i+1)*samplePerClass,i]=1    
        else:#repeat the labeling
            y[i*samplePerClass:(i+1)*samplePerClass,int(i-nClusters*0.5)]=1   
        trueProb[:,i]=pdfs[i].pdf(x)
    trueProb=normalize(trueProb,norm='l1') 
    for i in range(x.shape[0]):
        for j in range(len(pdfs)):
            if(trueProb[i,j]>thres):
                if(j<int(0.5*nClusters)):
                    y[i,j]=1
                else:
                    y[i,int(j-nClusters*0.5)]=1
    print(y.shape)
    def labelCluster(clusterList,labelList):
        return
    if(additionalRule):
        #randomly pick 10% data to generate 2 new labels
        cardi=np.sum(y,axis=1)
        #add labels indicating cardi=2 and cardi=4
        y=np.concatenate([y,np.zeros([y.shape[0],2])],axis=1)
        for i in range(y.shape[0]):
            if(cardi[i]==3):
                y[i,-2]=1
            elif(cardi[i]>3):
                y[i,-1]=1
        #add two logical labels wrt l1 l2, l2 l3,  labels(and or opertation)
        y=np.concatenate([y,np.zeros([y.shape[0],8])],axis=1)
        for i in range(y.shape[0]):
            if(y[i,0] and y[i,1]): #over lapping zone label. 
                y[i,-8]=1
            if(y[i,1] and y[i,2]): # over lapping zone label
                y[i,-7]=1
            if(y[i,2] and y[i,3]): #over lapping zone label. 
                y[i,-6]=1
            if(y[i,4] and y[i,5]): # over lapping zone label
                y[i,-5]=1
        
        tem=y[:,0]
        for i in [5]:
            tem=np.logical_or(tem,y[:,i])#index of data with label 0,5
        index=np.where(tem==1)[0]
        np.random.shuffle(index)
        #the 'contain' relationship
        y[index,-4]=1 #data with label 0 or 5 has this label -5
        y[index[0:int(0.3*len(index))],-3]=1#1/3 of the data with label -5 has label -6
        np.random.shuffle(index)
        y[index[0:int(0.3*len(index))],-2]=1 #1/3 of the data with label -5 has label -7        
        #exclusive relationship
        print('what')
        for i in range(y.shape[0]):
            if(y[i,-2] and not y[i,-3]):
                y[i,-1]=1
    print('label sparsity:%f'%(np.sum(y)/(y.shape[0]*y.shape[1])))
    print('number of instance cardi=2: %f cardi=3 %f cardi=4 %f cardi=5 %f' %(len(np.where(np.sum(y,axis=1)==2)[0]),len(np.where(np.sum(y,axis=1)==3)[0]),len(np.where(np.sum(y,axis=1)==4)[0]),len(np.where(np.sum(y,axis=1)==5)[0])))
    return x,y,pdfs,clusterIndex,trueProb
'''