#CNN model
import torch
from torch import nn
from layers import *
from config import *

#不是每个都有的,也有多出来的
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
win_width=config['win_width']
shape2=config['shape2']
block_num=config['block_num']
input_chans=config['input_chans']

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        block = []
        for i in range(0, block_num[2]):
            num_features = num_features2
            if i == 0:
                block.append(preblock(input_chans, filter_size1,num_features1,pooling_size))  # intialization!!!!
            else:
                block.append(preblock(num_features1, filter_size1, num_features1, pooling_size))
        self.preblock_block = nn.Sequential(*block)

        block = []
        for i in range(0,block_num[1]):
            if i==0:
                block.append(ResCLSTM(shape2, num_features1, filter_size2,num_features2,batch_size,num_layers=2)) #intialization!!!!
            else:
                block.append(ResCLSTM(shape2, num_features2, filter_size2, num_features2, batch_size, num_layers=2))
        self.ResCLSTM_block=nn.Sequential(*block)

        block = []
        if block_num[2]!=1:
            for i in range(0,block_num[2]):
                num_features=num_features2
                if i==0:
                    block.append(NiN(shape2, num_features2, filter_size2,num_features3,batch_size)) #intialization!!!!
                elif i==block_num[2]-1:
                    block.append(NiN(shape2, num_features3, filter_size2, num_features3, batch_size,last_AF=False))
                else:
                    block.append(NiN(shape2, num_features3, filter_size2, num_features3, batch_size))
        else:
            block.append(NiN(shape2, num_features2, filter_size2, num_features3, batch_size, last_AF=False))
        self.NiN_block=nn.Sequential(*block)
        #self.convlstm=CLSTM(shape2,input_chans=num_features3, filter_size=filter_size2,
        #                          num_features=n_note,num_layers=1,batch_size=batch_size,padding_mode='same',input_form='BCSWH') no this layer in CT1
        #a special form convlstm=fully connected lstm,no need to write a new one and add a reshape

    def forward(self, x):

        # # (8L, 1L, 38L, 252L) -> (8L, 32 L, 1L, 7L, 252L)  :  conv input -> convlstm input valid mode -> convlstm input same mode
        # _,x = self.convlstm1(x,self.hidden_state1)#(8L, 50L, 34L, 228L) -> (8L, 32 L, 50L, 3L, 228L) -> (8L, 32 L, 50L, 7L, 252L)
        # x = self.maxpool(x)#(8L, 50L, 34L, 76L) -> (8L, 32 L, 50L, 3L, 76L) -> (8L, 32 L, 50L, 7L, 84L)
        # _,x = self.convlstm2(x,self.hidden_state2)  #(8L, 50L, 32L, 72L) -> (8L, 32 L, 50L, 1L, 72L)  -> (8L, 32 L, 50L, 7L, 84L)
        # x = self.maxpool(x)#(8L, 50L, 32L, 24L)  -> (8L, 32 L, 50L, 1L, 24L)   -> (8L, 32 L, 50L, 1L, 28L)
        # if padding_mode=='valid':
        #     x = x.squeeze().view(batch_size,num_features2,win_width,filter_size3[1]) # batch size must be taken into account.
        # else:
        #     x = x.view(batch_size,num_features2,win_width,filter_size3[1])
        # x = self.conv3(x) # (8L, 1000L, 32L, 1L)
        # x = self.conv4(x) # (8L, 500L, 32L, 1L)
        # x = self.conv5(x) # (8L, 88L, 32L, 1L)
        # #print x.shape
        x=self.preblock_block(x)  #(B,1,seq_len,W,H)
        x=self.ResCLSTM_block(x)
        x=self.NiN_block(x)
        #x=self.convlstm(x)
        x=x.squeeze() #last sigmoid is in loss layer
        return x


def get_model():
    net = Net()
    loss = Loss() #config['num_hard'])
    return net, loss
