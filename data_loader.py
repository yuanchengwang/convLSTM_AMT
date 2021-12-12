import numpy as np
#import random
import os,glob
from torch.utils.data import Dataset
#import time
import torch
from data_prepare.onset_utils import *
from config import * # Aug_config is imported, not a parameter
#from multiprocessing import Pool
#from functools import partial
import sys
sys.path.append("./data_prepare")
import onset_utils as ou

# win_width=100
# kernel_size=7  #7*252=42**2=1764
# data_dir='/home/wyc/Desktop/preprocessed_data'
# (2L, 1L, 7L, 252L)
# (16L, 50L, 7L, 252L)




class data_loader(Dataset):
    def __init__(self, data_dir, win_width,input_chans, kernel_size,phase='train',model_path='conv',dual_train='Frame',drop_last=True):
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        assert (model_path == 'conv' or model_path == 'convlstm1' or model_path == 'convlstm2' or model_path == 'convlstm3' or model_path == 'conv_lstm')
        assert (dual_train == 'Frame' or dual_train == 'Onset' or  dual_train == 'Both')
        self.phase = phase
        filelist= glob.glob(os.path.join(data_dir,phase)+'/*')
        CQT_name=[f for f in filelist if (f[-7:-4]=='CQT')]
        self.input=[]
        self.nb_sample=[]
        self.label = []
        self.model_path=model_path
        self.win_width=win_width
        self.input_chans=input_chans
        self.kernel_size=kernel_size
        self.dual_train=dual_train
        self.drop_last=drop_last # use only test phase


        for i in CQT_name:
            if i[-3:]=='npz':
                self.input.append(cut(np.load(i)['arr_0'],win_width,kernel_size,padding=True,overlap_rate=0,dual_train=self.dual_train,drop_last=drop_last))   # 64s,no need to paralellize, I/O is limited ,246s with 5 workers
            else:
                self.input.append(cut(np.load(i), win_width, kernel_size, padding=True,overlap_rate=0,dual_train=self.dual_train,drop_last=drop_last))
            self.nb_sample.append(self.input[-1].shape[0])

        if self.phase != 'test':   #Attention: we have the ground truth for testset
            label_name=[f[:-7]+'label.npz' for f in CQT_name if os.path.exists(f[:-7]+'label.npz')]
            if label_name==[]:
                label_name=[f[:-7]+'label.npy' for f in CQT_name] # add the support for npy and npz
                for i in label_name:
                    self.label.append(cut(np.load(i)[:,:,88:],win_width,kernel_size,padding=False,overlap_rate=0,dual_train=self.dual_train,drop_last=drop_last).transpose(0,3,2,1)) #(B,W,H)->(B,H,W,1)  (B,1,W,H)->(B,H,W,1)
            else:
                for i in label_name:
                    self.label.append(cut(np.load(i)['arr_0'][:,:,88:], win_width, kernel_size, padding=False,overlap_rate=0,dual_train=self.dual_train,drop_last=drop_last).transpose(0,3,2,1))

    def __getitem__(self,idx):
        nb_list, sub_nb = index(idx, self.nb_sample)
        if self.phase!='test':  #no real test in our senerios
            if self.model_path=='conv' or self.model_path=='conv_lstm': #########################
                x=self.input[nb_list][sub_nb]
                y=self.label[nb_list][sub_nb]
                if self.phase=='train':
                    x,y=augment(x,y)
                return torch.from_numpy(x.astype(np.float32)),torch.from_numpy(y.astype(np.float32))
            #(1,106,252)/(1,100,88)
            elif self.model_path=='convlstm1'or self.model_path=='convlstm3' :
                x=self.input[nb_list][sub_nb]
                y = self.label[nb_list][sub_nb]
                if self.phase=='train':
                    x,y=augment(x,y)
                x1=np.zeros([self.win_width,self.input_chans,self.kernel_size, x.shape[-1]])
                for i in range(self.win_width):
                    x1[i,0,:,:]=x[0,i:(i+self.kernel_size),:] # the first dim will be ignored
#                print x1.shape
#                print y.shape
                return torch.from_numpy(x1.astype(np.float32)),torch.from_numpy(y.astype(np.float32))   #(100,1,7,252)/(1,100,88)
            else:# convlstm2 mode
                x = self.input[nb_list][sub_nb]
                y = self.label[nb_list][sub_nb]
                if self.phase=='train':
                    x,y=augment(x,y)
                x1 = np.zeros([1, self.win_width, self.kernel_size, x.shape[-1]])
                for i in range(self.win_width):
                    x1[0,i, :, :] = x[0, i:(i + self.kernel_size), :]  # the first dim will be ignored
#                    print x.shape
                return torch.from_numpy(x1.astype(np.float32)), torch.from_numpy(y.astype(np.float32)) #(1,100,7,252)/(1,100,88)

        else:    #no real test in our senerios,test phase
            if self.model_path=='conv'or self.model_path=='conv_lstm':
                    return torch.from_numpy(self.input[nb_list][sub_nb].astype(np.float32))
            #(1,106,252)/(1,100,88)
            elif self.model_path=='convlstm1'or self.model_path=='convlstm3':
                if self.drop_last:
                    x=self.input[nb_list][sub_nb]
                    x1 = np.zeros([x.shape[1] - self.kernel_size + 1, self.input_chans, self.kernel_size, x.shape[-1]])
                    for i in range(x.shape[1]-self.kernel_size+1):
                        x1[i,0,:,:]=x[0,i:(i+self.kernel_size),:]
                else:
                    x = self.input[nb_list][sub_nb][0]
                    x1 = np.zeros([x.shape[0] - self.kernel_size + 1, self.input_chans, self.kernel_size, x.shape[-1]])
                    for i in range(x.shape[0]-self.kernel_size+1):
                        x1[i,0,:,:]=x[i:(i+self.kernel_size),:]
            # print x1.shape
                return torch.from_numpy(x1.astype(np.float32))
            else:# convlstm2 mode
                if self.drop_last:
                    x = self.input[nb_list][sub_nb]
                    x1 = np.zeros([1,x.shape[1] - self.kernel_size + 1, self.kernel_size, x.shape[-1]])
                    for i in range(x.shape[1] - self.kernel_size + 1):
                        x1[0, i, :, :] = x[0, i:(i + self.kernel_size), :] #seq_len becomes the third dim,i.e. (B,1,seq_len,W,H)
                else:
                    x = self.input[nb_list][sub_nb][0]
                    x1 = np.zeros([1,x.shape[0] - self.kernel_size + 1, self.kernel_size, x.shape[-1]])
                    for i in range(x.shape[0] - self.kernel_size + 1):
                        x1[0, i, :, :] = x[i:(i + self.kernel_size), :] #seq_len becomes the third dim,i.e. (B,1,seq_len,W,H)
                        # print x1.shape
                return torch.from_numpy(x1.astype(np.float32)) #(1,100,7,252)

    def __len__(self):
        return sum(self.nb_sample)

def cut(matrix,win_width,kernel_size,padding=True,overlap_rate=0,dual_train='Frame',drop_last=True,onset_mode='concat',matrix_2='None',output_mode='concat'):  #window cut module
# cut the tensor along the first(time) axis by the win_width with a single frame hop
# padding=True for x generation, =False for y generation
# drop_last=False can be used since the seq_lens are the same in a mini-batch or batch_size=1
# Matrix_2 is the onset label matrix
    assert(overlap_rate < 1 and overlap_rate >= 0)
    assert(onset_mode=='stack' or onset_mode=='concat' or onset_mode=='isolated' or onset_mode=='None') #'None'= without onset matrix input
    assert(output_mode=='stack' or output_mode=='concat')
    l=matrix.shape[1]
    cut_matrix=[]
    hop_len = int(np.ceil(win_width*(1-overlap_rate))) 
    nb_hop=(l-win_width)/hop_len+1   #integer division=floor
    residu=l-win_width-(nb_hop-1)*hop_len

    if not dual_train == 'Both' or onset_mode=='stack' or onset_mode=='concat': # no dual train or dual_train with concatenate or stack mode
        if not padding:
            for i in xrange(nb_hop):
                cut_matrix.append(matrix[:,i*hop_len:(i*hop_len + win_width),:])
            if not drop_last and residu != 0:
                cut_matrix.append(matrix[:,l-residu:, :])
        else:
            w1=matrix.shape[0]
            w2=matrix.shape[2]
            matrix_1=np.concatenate([np.zeros([w1,kernel_size/2,w2]),matrix,np.zeros([w1,kernel_size/2,w2])],axis=1)  #padding
            for i in xrange(nb_hop):
                cut_matrix.append(matrix_1[:,i*hop_len:i*hop_len+win_width+kernel_size-1,:])    #0-104,100-204,...
#            print cut_matrix[0].shape
            if not drop_last and residu != 0:
                cut_matrix.append(matrix_1[:,l - residu:, :])
    elif dual_train == 'Both' and (onset_mode=='isolated' or onset_mode=='None'):# dual training mode, onset_mode
        if onset_mode=='None' or matrix_2=='None': #One of the onset_mode and matrix_2 not defined will lead to an onset matrix generation
            matrix_2=ou.onset_diff(matrix)            # Only onset_mode=='isolated' and matrix_2 exists won't generate an onset matrix
        if output_mode == 'stack':
            axis = 0
        else:#concat
            axis = 2

        if not padding:
            for i in xrange(nb_hop):
                cut_matrix.append(np.concatenate([matrix[:,i * hop_len:(i * hop_len + win_width), :],matrix_2[:,i * hop_len:(i * hop_len + win_width), :]],axis=axis)) #-1
            if not drop_last and residu != 0:
                cut_matrix.append(np.concatenate([matrix[:,l-residu:, :],
                                              matrix_2[:,l-residu:, :]],axis=axis))  # -1
        else:
            w1 = matrix.shape[0]
            w2 = matrix.shape[2]
            matrix_1 = np.concatenate([np.zeros([w1,kernel_size / 2, w2]), matrix, np.zeros([w1,kernel_size / 2, w2])],
                                      axis=1)  # padding
            matrix_3 = np.concatenate([np.zeros([w1,kernel_size / 2, w2]), matrix_2, np.zeros([w1,kernel_size / 2, w2])],
                                      axis=1)

            for i in xrange(nb_hop):
                cut_matrix.append(np.concatenate([matrix_1[:,i * hop_len:i * hop_len + win_width + kernel_size - 1, :],matrix_3[:,i * hop_len:i * hop_len + win_width + kernel_size - 1, :]],axis=axis))  # 0-104,100-204,...
            if not drop_last and residu!=0:
                cut_matrix.append(np.concatenate([matrix_1[:,l-residu:, :],
                                              matrix_3[:,l-residu:, :]],axis=axis))   #-1
    cut_matrix = np.asarray(cut_matrix) #output_mode=stack
    return cut_matrix



def index(idx,nb_sample):
    l=len(nb_sample)
    accum_nb =0
    nb_list=0
    sub_nb=0
    for i in range(l):
        accum_nb+=nb_sample[i]
        if idx < accum_nb:
            nb_list, sub_nb= i, idx+nb_sample[i]-accum_nb
            break
    return nb_list,sub_nb

def augment(x,y,split=False):
    if Aug_config['amp_aug']:   #without label changement
        x = amplitude_aug(x, Aug_config['amp_range'])
    if Aug_config['key_aug']:
        x,y= key_aug(x,y,Aug_config['key_range'],split=split) # If there is a redundancy frequency, split can be used!
    if Aug_config['pitch_shift']: #without label changement
        x = pitch_shifting(x, Aug_config['pitch_shift_range'])
    return x,y

def amplitude_aug(x,range):
    amp=10**(np.random.uniform(range[0],range[1])/10) #Continue value, unity (db)
    return x*amp


def label_range(y): #y.shape=(88,32,1)
    y=np.sum(y[:config['n_note'],:,0],axis=1) # 0:88, no matter there is the onset or not
    y=np.where(y>0)[0]
    try:
        return [min(y),max(y)]
    except:
        return []
    
def key_aug(x,y,range,split=False):
    assert(range[0]<=0 and range[1]>=0)
    labelrange=label_range(y)
    if labelrange==[]:
        return x,y
    k = np.random.randint(max(range[0],-labelrange[0]), min(range[1] + 1,config['n_note']-labelrange[1])) #random with limite
    n=Aug_config['bins_per_octave']/12
    x=pitch_shifting(x,k*n,random=False)#shift bins for a semitone
    y=pitch_shifting(y,k,random=False,axis=0)
    if split:
        x=x[:,:,(-range[0]+k)*n-1:(-range[1]+k)*n] # in case of giving more frequency bins for input x
    return x,y

def pitch_shifting(x,range,random=True,axis=2):
    if random:
        range=np.random.randint(range[0],range[1]+1)
    if range==0:
        return x
    x_new = np.zeros(x.shape)
    if axis==2:
        if range>0:#shift up
            x_new[:,:,range:]=x[:,:,:-range] #channel,win_width,n_freq
        else:#shift down
            x_new[:, :, :-range] = x[:, :, range:]
    if axis==0:
        if range>0:#shift up
            x_new[range:,:,:]=x[:-range,:,:] #channel,win_width,n_freq
        else:#shift down
            x_new[:-range, :, :] = x[range:, :, :]
    return x_new
