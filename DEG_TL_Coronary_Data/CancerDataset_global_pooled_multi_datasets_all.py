#DEGnext
#Copyright Tulika Kakati, 2020
#Distributed under GPL v3 license
#This code may be used to import cancer RNA-seq dataset(s) into Pytorch DataLoader

from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import Normalizer
import torch
import os
dir_path = os.path.dirname(os.path.abspath(os.curdir))
#os.path.abspath(os.curdir) #changed path refer to github for original for vscode


class MyDataset(Dataset):
    """ My dataset."""

    # Initialize your data, download, etc.
    def __init__(self,xy=None,xy_label=None,transform=None,case=None,data_type=None):
        print('loading dataset')
        if (case==None): 
            if (is_prime(xy.shape[1])==True):
                M=xy.shape[0]
                N=xy.shape[1]
                b = np.zeros((M,N+1))
                b[:,:-1] = xy
                xy=b
            N=xy.shape[0]
            noise_factor=1
            noise_data = np.random.randn(xy.shape[0],xy.shape[1])*noise_factor
            new_data=xy + noise_data
            xy=new_data
            self.transform=transform
            self.x_data=xy
            self.len=self.x_data.shape[0]
            self.y_data = xy_label
        
        if(case=='train'):
            print(" single train dataset loading")

            File = dir_path+'/datasets/'+data_type+'_non_bio_train.csv'
            xy = np.loadtxt(File,delimiter=',', dtype=np.float32)
            File = dir_path+'/datasets/'+data_type+'_non_bio_train_label.txt'
            xy_label = np.loadtxt(File,delimiter='\t', dtype=np.float32)
            if(is_prime(xy.shape[1])==True):
                    M=xy.shape[0]
                    N=xy.shape[1]
                    b = np.zeros((M,N+1))
                    b[:,:-1] = xy
                    xy=b

            N=xy.shape[0]
            noise_factor=1
            noise_data = np.random.randn(xy.shape[0],xy.shape[1])*noise_factor
            new_data=xy + noise_data
            xy=new_data
            self.transform=transform
            self.x_data=xy
            self.len=self.x_data.shape[0]
            self.y_data = xy_label

        if(case=='testing'):

            print(" Pig Test dataset loading")
            File = dir_path+'/datasets/'+data_type+'_test.csv'
            xy = np.loadtxt(File,delimiter=',', dtype=np.float32)
            File = dir_path+'/datasets/'+data_type+'_test_label.txt'
            xy_label = np.loadtxt(File,delimiter='\t', dtype=np.float32)
            if(is_prime(xy.shape[1])==True):
                    M=xy.shape[0]
                    N=xy.shape[1]
                    b = np.zeros((M,N+1))
                    b[:,:-1] = xy
                    xy=b

            N=xy.shape[0]
            noise_factor=1
            noise_data = np.random.randn(xy.shape[0],xy.shape[1])*noise_factor
            new_data=xy + noise_data
            xy=new_data
            self.transform=transform
            self.x_data=xy
            self.len=self.x_data.shape[0]
            self.y_data = xy_label


    def __getitem__(self, index):
        return self.transform(self.x_data[index]),self.y_data[index]

    def __len__(self):
        return self.len

def is_prime(a):
    return all(a % i for i in range(2, a))



