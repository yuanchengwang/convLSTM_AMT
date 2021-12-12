 #CNN model
import torch
from torch import nn
from layers import *
from config import *


input_chans=config['input_chans']
pooling_size=config['pooling_size']
filter_size1=config['filter_size1']
filter_size2=config['filter_size2']
filter_size3=config['filter_size3']
num_features1=config['num_features1'] #50
num_features2=config['num_features2'] #50
num_features3=config['num_features3'] #1000
num_features4=config['num_features4']
n_note=config['n_note'] #88
if config['padding_mode']=='same':
    if type(filter_size1)==tuple:  # filter_size may not be a square
        padding1 = ((filter_size1[0] - 1) / 2, (filter_size1[1] - 1) / 2)
    else:  # filter size = number
        padding1 = (filter_size1 - 1) / 2 #became a tuple
    if type(filter_size2)==tuple:  # filter_size may not be a square
        padding2 = ((filter_size2[0] - 1) / 2, (filter_size2[1] - 1) / 2)
    else:  # filter size = number
        padding2 = (filter_size2 - 1) / 2 #became a tuple    

elif config['padding_mode']=='valid':
    padding1=0
    padding2=0


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_size, return_indices=False)

        self.conv1= nn.Sequential(
            nn.Conv2d(input_chans, num_features1, kernel_size=filter_size1, stride=1,padding=padding1),
            nn.BatchNorm2d(num_features1),
            nn.ELU(inplace=True))
            #nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_features1, num_features2, kernel_size=filter_size2, stride=1,padding=padding2),
            nn.BatchNorm2d(num_features2),
            nn.ELU(inplace=True))
        # nn.ReLU(inplace=True)
        self.conv3=nn.Sequential(
            nn.Conv2d(num_features2, num_features3, kernel_size=filter_size3),   #kernel_size !!!!! conv_mode,padding=0 by default
            #nn.ReLU(inplace=True),
            nn.ELU(inplace=True))#,
            #nn.Dropout3d(p=0.5, inplace=False))
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_features3, num_features4, kernel_size=1),  # kernel_size !!!!! conv_mode,padding=0 by default
            nn.ELU(inplace=True))#,
        #nn.ReLU(inplace=True),
            #nn.Dropout3d(p=0.5, inplace=False))
        self.conv5=nn.Sequential(
            nn.Conv2d(num_features4, n_note, kernel_size=1))#,
            #nn.Sigmoid())

    def forward(self, x):
        # (8L, 1L, 38L, 252L)
        #print x.shape
        x = self.conv1(x)#(8L, 50L, 34L, 228L)   246L
        #print x.shape
        x = self.maxpool(x)#(8L, 50L, 34L, 76L)  82L
        x = self.conv2(x)  #(8L, 50L, 32L, 72L)  78L
        x = self.maxpool(x)#(8L, 50L, 32L, 24L)  26L
        
        x = self.conv3(x)# (8L, 1000L, 32L, 1L)
        x = self.conv4(x)#(8L, 500L, 32L, 1L)
        #print x.shape
        x = self.conv5(x)#(8L, 88L, 32L, 1L)
        #print x.shape
        return x


def get_model():
    net = Net()
    loss = Loss()#config['num_hard'])
    return net, loss