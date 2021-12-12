import torch
from torch import nn
from torch.nn import LSTM
from layers import *
from config import *


batch_size=config['batch_size']
win_width = config['win_width']
padding_mode = config['padding_mode']
pooling_size=config['pooling_size']
peephole=config['peephole']
Layer_norm=config['Layer_norm']
filter_size1=config['filter_size1']
filter_size2=config['filter_size2']
filter_size3=config['filter_size3']
num_features1=config['num_features1'] #50
num_features2=config['num_features2'] #50
num_features3=config['num_features3'] #1000
num_features4=config['num_features4'] #500
n_note=config['n_note'] #88
kernel_size=config['kernel_size']
n_freq=config['n_freq']
#win_width=config['win_width']
input_chans=config['input_chans']

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
        self.maxpool = nn.MaxPool3d(kernel_size=pooling_size, stride=pooling_size, return_indices=False) # the parameters must be odd
  
        self.conv1=nn.Sequential(
            nn.Conv2d(1, num_features1, kernel_size=filter_size1,padding = padding1),   
            nn.ELU(inplace=True),
            nn.Dropout3d(p=0.5, inplace=False))
        self.conv2=nn.Sequential(
            nn.Conv2d(num_features1, num_features2, kernel_size=filter_size2,padding = padding2),   
            nn.ELU(inplace=True),
            nn.Dropout3d(p=0.5, inplace=False)) 
        self.lstm3 = LSTM(input_size = num_features2*24,hidden_size = num_features3)   # ,dropout = 0.5
        self.lstm4 = LSTM(input_size = num_features3,hidden_size = num_features4)     #,dropout = 0.5    
        self.conv5=nn.Sequential(
            nn.Conv2d(num_features4, n_note, kernel_size=1))


    def forward(self, x):
        # (8L, 1L, 38L, 252L) -> (8L, 32 L, 1L, 7L, 252L)  :  conv input -> convlstm input valid mode -> convlstm input same mode
#        print '1',x.shape
        x = self.conv1(x)
#        print '2',x.shape
        x = self.maxpool(x)
#        print '3',x.shape        
        x = self.conv2(x)  
#        print '4',x.shape
        x = self.maxpool(x)
#        print '5',x.shape
        x = x.transpose(0,2)
        x = x.transpose(1,2)
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3])    
#        print '6',x.shape   
        x,_ = self.lstm3(x)
#        print '7',x.shape
        x,_ = self.lstm4(x) 
#        print '8',x.shape
        x = x.transpose(0,1)
        x = x.transpose(1,2)
#        print '8.5',x.shape
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2],-1)
#        print '9',x.shape
        x = self.conv5(x) 
#        print '10',x.shape
        return x


def get_model():
    net = Net()
    loss = Loss() #config['num_hard'])
    return net, loss