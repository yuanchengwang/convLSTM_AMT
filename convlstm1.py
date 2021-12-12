#CNN model
import torch
from torch import nn
from layers import *
from config import *

batch_size=config['batch_size']
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
shape2=config['shape2']
input_chans=config['input_chans']


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=pooling_size, stride=pooling_size, return_indices=False) # the parameters must be odd

        self.convlstm1= CLSTM(shape=(kernel_size,n_freq),input_chans=input_chans, filter_size=filter_size1, num_features=num_features1,num_layers=1,batch_size=batch_size,padding_mode=padding_mode) #,
        #shape,input_channel,filter_size,feature_number,layer_number
        self.convlstm1.apply(weights_init) # Initialize the weight
        self.hidden_state1 = self.convlstm1.init_hidden() # Initialize the h0,c0
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(50, 50, kernel_size=(3,5), stride=1),
        #     nn.BatchNorm2d(50),
        #     nn.ELU(inplace=True))
        self.convlstm2 = CLSTM(shape=shape2, input_chans=num_features1, filter_size=filter_size2, num_features=num_features2, num_layers=1,
                               batch_size=batch_size,peephole=peephole,Layer_Normalization=Layer_norm,padding_mode=padding_mode)
        # shape,input_channel,filter_size,feature_number,layer_number

        # need a flatten/reshape layer? no need, because the input is no more an entire plane
        self.convlstm2.apply(weights_init)
        self.hidden_state2 = self.convlstm2.init_hidden()
        # print self.hidden_state2[0][0].shape
        # nn.ReLU(inplace=True)

        # need a flatten/reshape layer,yes see the forward block
        self.conv3=nn.Sequential(
            nn.Conv2d(num_features2, num_features3, kernel_size=filter_size3),   #kernel_size !!!!! conv_mode,padding=0 by default
            #nn.ReLU(inplace=True),
            nn.ELU(inplace=True),
            nn.Dropout3d(p=0.5, inplace=False))
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_features3, num_features4, kernel_size=1),  # kernel_size !!!!! conv_mode,padding=0 by default
            nn.ELU(inplace=True),
        #nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.5, inplace=False))
        self.conv5=nn.Sequential(
            nn.Conv2d(num_features4, n_note, kernel_size=1))#,
            #nn.Sigmoid())

    def forward(self, x):
        # (8L, 1L, 38L, 252L) -> (8L, 32 L, 1L, 7L, 252L)  :  conv input -> convlstm input valid mode -> convlstm input same mode
#        print x.shape
        _,x = self.convlstm1(x,self.hidden_state1)#(8L, 50L, 34L, 228L) -> (8L, 32 L, 50L, 3L, 228L) -> (8L, 32 L, 50L, 7L, 252L)
#        print x.shape
        x = self.maxpool(x)#(8L, 50L, 34L, 76L) -> (8L, 32 L, 50L, 3L, 76L) -> (8L, 32 L, 50L, 7L, 84L)
        _,x = self.convlstm2(x,self.hidden_state2)  #(8L, 50L, 32L, 72L) -> (8L, 32 L, 50L, 1L, 72L)  -> (8L, 32 L, 50L, 7L, 84L)
        x = self.maxpool(x)#(8L, 50L, 32L, 24L)  -> (8L, 32 L, 50L, 1L, 24L)   -> (8L, 32 L, 50L, 1L, 28L)
        if padding_mode=='valid':
            x = x.squeeze().view(batch_size,num_features2,-1,filter_size3[1]) # batch size must be taken into account.
        else:
            x = x.view(batch_size,num_features2,-1,filter_size3[1]) # win_width is replaced by -1 to adapt different input seq_len
        x = self.conv3(x) # (8L, 1000L, 32L, 1L)
        x = self.conv4(x) # (8L, 500L, 32L, 1L)
        x = self.conv5(x) # (8L, 88L, 32L, 1L)
        #print x.shape
        return x


def get_model():
    net = Net()
    loss = Loss() #config['num_hard'])
    return net, loss