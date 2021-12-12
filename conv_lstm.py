import torch
from torch import nn
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
        self.lstm3= LSTM(shape=(1,1),input_chans = num_features2*24, filter_size=filter_size1, num_features=num_features3,num_layers=1,batch_size=batch_size) #,
        #shape,input_channel,filter_size,feature_number,layer_number
        self.lstm3.apply(weights_init) # Initialize the weight
        self.hidden_state3 = self.lstm3.init_hidden() # Initialize the h0,c0
        
        self.lstm4= LSTM(shape=(1,1),input_chans = num_features3, filter_size=filter_size1, num_features=num_features4,num_layers=1,batch_size=batch_size) #,
        #shape,input_channel,filter_size,feature_number,layer_number
        self.lstm4.apply(weights_init) # Initialize the weight
        self.hidden_state4 = self.lstm4.init_hidden() # Initialize the h0,c0 
        self.conv5=nn.Sequential(
            nn.Conv2d(num_features4, n_note, kernel_size=1))


    def forward(self, x):
        #x.shape = [16, 1, 22, 252]
        x = self.conv1(x)#[16, 50, 18, 228]
#        print '2',x.shape
        x = self.maxpool(x)#[16, 50, 18, 76]
#        print '3',x.shape        
        x = self.conv2(x)  #[16, 50, 16, 72]
#        print '4',x.shape
        x = self.maxpool(x)#[16, 50, 16, 24]
#        print '5',x.shape
        x = x.transpose(1,2)
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3],1,1)    #[16, 1, 1200, 16, 1]
#        print '6',x.shape
        _,x = self.lstm3(x,self.hidden_state3) #[16, 1, 1000, 16, 1]
#        print '7',x.shape
        _,x = self.lstm4(x,self.hidden_state4) #[16, 1, 500, 16, 1]
#        print '8',x.shape        x.shape=[B,seq_len,Chan,1,1]
        x = x.squeeze(dim=4).transpose(1,2)
#        print '9',x.shape
        x = self.conv5(x) #[16, 500, 16, 1]
#        print '10',x.shape
        return x


def get_model():
    net = Net()
    loss = Loss() #config['num_hard'])
    return net, loss