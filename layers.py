# coding=utf-8
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import math
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
dev=torch.device('cuda')

#def hard_mining(neg_output, neg_labels, num_hard):
#    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
#    neg_output = torch.index_select(neg_output, 0, idcs)
#    neg_labels = torch.index_select(neg_labels, 0, idcs)
#    return neg_output, neg_labels


class Loss(nn.Module):
    def __init__(self, num_hard=0):
        super(Loss, self).__init__()
        #self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCEWithLogitsLoss() #weight is defined for samples
        #self.num_hard = num_hard

    def forward(self, output, labels, dual_train='Frame'):
        #batch_size = labels.size(0)
        #output = torch.transpose(output,0,3,2,1) #[6,88,32,1]->[4224,4]
        #output=output.view(-1,88)
        #labels = labels.view(-1,88)
        #[6,1,32,88]->[4224,4] unbalance
        
#        loss = self.classify_loss(output,labels) #######
        if dual_train == 'Frame':
            loss = self.classify_loss(output,labels) 
            pos = (torch.sigmoid(output) >= 0.5).type(torch.cuda.FloatTensor)  #it works 
            recall=labels.sum()
            precision=pos.sum()
            TP = (pos*labels).sum()
            return [loss,TP.item(), precision.item(), recall.item()] #F-score must be computed by whole epoch
        else:#onset occupies latter half of the output dim
            #loss
            loss_total = self.classify_loss(output,labels)  
            if dual_train == 'Both':
                loss_frame = self.classify_loss(output[:,:output.shape[1]/2,:,:],labels[:,:output.shape[1]/2,:,:])
                loss_onset = self.classify_loss(output[:,output.shape[1]/2:,:,:],labels[:,output.shape[1]/2:,:,:])
                # frame P/R/F      
                pos_output_frame = (torch.sigmoid(output[:,:output.shape[1]/2,:,:]) >= 0.5).type(torch.cuda.FloatTensor)    #dimension may not right
                pos_label_frame = labels[:,:output.shape[1]/2,:,:]
                recall_frame = pos_label_frame.sum()
                precision_frame = pos_output_frame.sum()
                TP_frame = (pos_output_frame*pos_label_frame).sum()
                # onset P/R/F
                pos_label_onset = (labels[:,output.shape[1]/2:,:,:] >= 0.9).type(torch.cuda.FloatTensor)
                pos_output_onset = (torch.sigmoid(output[:,output.shape[1]/2:,:,:]) >= 0.5).type(torch.cuda.FloatTensor) # P/R/F are only defined for o-1 matrix  
                pos_label_onset2 = (labels[:,output.shape[1]/2:,:,:] >= 0.5).type(torch.cuda.FloatTensor)
            else:
                pos_label_onset = (labels >= 0.9).type(torch.cuda.FloatTensor)
                pos_output_onset = (torch.sigmoid(output) >= 0.5).type(torch.cuda.FloatTensor) # P/R/F are only defined for o-1 matrix
                pos_label_onset2 = (labels >= 0.5).type(torch.cuda.FloatTensor)
            pos_tp_onset = (pos_output_onset*pos_label_onset2 > 0).type(torch.cuda.FloatTensor)   
            
            pos_output_onset = pos_output_onset.transpose(0,1).reshape(output.shape[1]/2,-1)
            pos_tp_onset = pos_tp_onset.transpose(0,1).reshape(output.shape[1]/2,-1)

            pos_output_diff = torch.zeros_like(pos_output_onset)      
            
            pos_tp_diff = torch.zeros_like(pos_tp_onset)
            
            for i in range(pos_output_diff.shape[1]):
                pos_output_diff[:,i] = pos_output_onset[:,i] - pos_output_onset[:,i-1]
                pos_tp_diff[:,i] = pos_tp_onset[:,i] - pos_tp_onset[:,i-1]
            pos_output_diff = (pos_output_diff > 0).type(torch.cuda.FloatTensor)
            pos_tp_diff = (pos_tp_diff > 0).type(torch.cuda.FloatTensor)
            
            recall_onset = pos_label_onset.sum()
            precision_onset = pos_output_diff.sum()
            TP_onset = pos_tp_diff.sum()             

            if dual_train == 'Both':            
                return [loss_total,[loss_frame.item(),loss_onset.item()],[TP_frame.item(),precision_frame.item(),recall_frame.item()],[TP_onset.item(),precision_onset.item(),recall_onset.item()]]
            else:
                return [loss_total,[TP_onset.item(),precision_onset.item(),recall_onset.item()]]


"""
LSTM implementation with layer normalization and peephole
Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
    Ba, Jimmy Lei, J. R. Kiros, and G. E. Hinton. "Layer Normalization." (2016).
    Felix A Gers, Nicol N Schraudolph, and Jürgen Schmidhuber. Learning precise timing with lstm recurrent networks. Journal of machine learning research,3(Aug):115–143, 2002.
"""

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:   #!=-1 fuzzy search, -1 means not found,find return the position index
        m.weight.data.normal_(0.0, 0.02)  # 'str in list' is absolute search form
    elif classname.find('BatchNorm') != -1:   # why? this kind of initialization ??????????
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    


class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """
    def __init__(self, shape, input_chans, filter_size, num_features,batch_size,Layer_Normalization=True,peephole=True,padding_mode='same'):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H,W
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.batch_size=batch_size
        self.padding_mode = padding_mode
        self.Layer_normalization = Layer_Normalization
        self.peephole=peephole

        if type(self.filter_size)==tuple:  # filter_size may not be a square
            self.padding = ((self.filter_size[0] - 1) / 2, (self.filter_size[1] - 1) / 2)
        else:  # filter size = number
            self.padding = ((self.filter_size - 1) / 2,(self.filter_size - 1) / 2) #became a tuple

        if self.Layer_normalization:
            if self.padding_mode=='same':
                self.LN=nn.LayerNorm([self.num_features,self.shape[0],self.shape[1]]) # no matter how many layers of LSTM
            elif self.padding_mode=='valid':
                self.LN = nn.LayerNorm([self.num_features, self.shape[0]-2*self.padding[0], self.shape[1]-2*self.padding[1]])

        if self.padding_mode=='same':
            # in this way the output has the same size
            # Conv2d argparameter:input channel+feature=input+hidden output_channel, kernel size, slide=1,padding=0 as default
            # [wx,wh]*[x,h]=x*wx+h*wh, note: the x size must be the same to the h size, if not use the following method
            self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
                              self.padding)
        elif self.padding_mode=='valid':
            self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
                                  0)  #hidden part must be zero-padded
            self.padding_layer=nn.ZeroPad2d((self.padding[1],self.padding[1],self.padding[0],self.padding[0]))

        if self.peephole:
            if self.padding_mode=='same':
                self.wi = Parameter(torch.Tensor(self.num_features, self.shape[0], self.shape[1]))#  Fully-Connected layer in paper is as same as CNN2d without flattening and extending
                self.wf = Parameter(torch.Tensor(self.num_features, self.shape[0], self.shape[1]))
                self.wo = Parameter(torch.Tensor(self.num_features, self.shape[0], self.shape[1]))
            else:
                self.wi = Parameter(torch.Tensor(self.num_features, self.shape[0]-2*self.padding[0], self.shape[1]-2*self.padding[1]))
                self.wf = Parameter(torch.Tensor(self.num_features, self.shape[0]-2*self.padding[0], self.shape[1]-2*self.padding[1]))
                self.wo = Parameter(torch.Tensor(self.num_features, self.shape[0]-2*self.padding[0], self.shape[1]-2*self.padding[1]))
            self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./ math.sqrt(sum(self.wi.size())) # what's the initializer?
        self.wi.data.uniform_(-stdv, stdv)
        self.wf.data.uniform_(-stdv, stdv)
        self.wo.data.uniform_(-stdv, stdv)


    def forward(self, input, hidden_state):
        hidden, c = hidden_state  # hidden and c are regarded as images with several channels
        # print 'hidden ',hidden.size()
        # print 'input ',input.size()
        #(16L, 50L, 3L, 228L) without padding,right size for hidden
        #(16L, 1L, 3L, 228L)
        #(16L, 50L, 3L, 76L) for input of second convlstm layer
        #print input.shape

        if self.padding_mode=='valid':
            hidden=self.padding_layer(hidden)
#        print 'hidden',hidden.shape
#        print 'input',input.shape
        combined = torch.cat((input, hidden), 1)  # concatenate in the channels
#        print self.input_chans + self.num_features
#        print 'combined',combined.size()
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)  # it should return 4 tensors, equally split

        if self.peephole:
            ai+=self.wi*c
            af+=self.wf*c

        if self.Layer_normalization:
            af = self.LN(af)
            #o = torch.sigmoid(self.LN(ao))
            ag = self.LN(ag)
            ai = self.LN(ai)

        i = torch.sigmoid(ai)
        f = torch.sigmoid(af+1.0) # WARNING:After layer normalization, the forget gate bias is eliminated, and its value is close to 0.5, +1 if LN
        #o = torch.sigmoid(ao)
        g = torch.tanh(ag)
        next_c = f * c + i * g

        if self.peephole:
            ao += self.wo*next_c

        if self.Layer_normalization:
            ao=self.LN(ao)
            next_c=self.LN(next_c)

        o=torch.sigmoid(ao)
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self):
        if self.padding_mode=='same':
            return (torch.zeros(self.batch_size, self.num_features, self.shape[0], self.shape[1]).to(dev),  #h0
                    torch.zeros(self.batch_size, self.num_features, self.shape[0], self.shape[1]).to(dev))  #c0  same size initialized by zeros
        elif self.padding_mode=='valid':
            return (torch.zeros(self.batch_size, self.num_features, self.shape[0]-2*self.padding[0], self.shape[1]-2*self.padding[1]).to(dev),
                torch.zeros(self.batch_size, self.num_features,self.shape[0]-2*self.padding[0], self.shape[1]-2*self.padding[1]).to(dev))


class CLSTM(nn.Module):
    """Initialize a basic Conv LSTM cell.
       for a whole unfold CLSTM(with len)
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """
    def __init__(self, shape, input_chans, filter_size, num_features, num_layers,padding_mode,batch_size,Layer_Normalization=True,peephole=True,input_form='BSCWH'):
        super(CLSTM, self).__init__()
        assert(input_form=='BSCWH' or input_form=='BCSWH')
        self.shape = shape  # H,W
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.num_layers = num_layers
        self.padding_mode=padding_mode
        self.batch_size=batch_size
        self.Layer_Normalization=Layer_Normalization
        self.peephole=peephole
        self.input_form=input_form
        cell_list = []
        cell_list.append(
            CLSTM_cell(self.shape, self.input_chans, self.filter_size, self.num_features,self.batch_size,self.Layer_Normalization,
                       self.peephole,self.padding_mode).to(dev))  # the first
        # one has a different number of input channels

        if padding_mode=='same': #multi-layer convlstm doesn't work in valid mode because of the different shape parameter
            for idcell in xrange(1, self.num_layers):
                cell_list.append(CLSTM_cell(self.shape, self.num_features, self.filter_size, self.num_features,self.batch_size,
                                            self.Layer_Normalization,self.peephole,self.padding_mode).to(dev))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input, hidden_state):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape seq_len,Batch,Chans,H,W

        """

        if self.input_form=='BCSWH':
            current_input = input.transpose(1, 2) #convert into the 'BSCWH'
        else:
            current_input = input      #[1,16,1,7,252]
        seq_len = current_input.size(1)
        # current_input = input.transpose(0, 1)  # now is seq_len,B,C,H,W, simplify the writing for current_input[t, ...],redundancy operation

        # current_input=input shape=(32L, 16L, 1L, 7L, 252L)
        next_hidden = []  # hidden states(h and c)

        for idlayer in xrange(self.num_layers):  # loop for every layer

            hidden_c = hidden_state[idlayer]  # hidden and c are images with several channels
            all_output = []
            output_inner = []
            for t in xrange(seq_len):  # loop for every step
                hidden_c = self.cell_list[idlayer](current_input[:,t, ...],
                                                   hidden_c)  # cell_list is a list with different conv_lstms 1 for every layer  
                output_inner.append(hidden_c[0]) #hidden_c[0]=hidden,hidden_c[1]=c
            next_hidden.append(hidden_c) #the last hidden c output
            current_input = torch.cat(output_inner, 1).view(output_inner[0].size(0),seq_len,
                                                            *output_inner[0].size()[1:])  #*output_inner[0].size() before
            #current_input = current_input.transpose(0, 1)
#            print 'current_input',current_input.shape
        if self.input_form == 'BCSWH': # back to this input form
            current_input=current_input.transpose(1,2)
        return next_hidden, current_input  # get back to the shape: B,seq_len,chans,H,W

    def init_hidden(self): # no relation with layer numbers
        init_states = []  # this is a list of tuples
        for i in xrange(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden())
        return init_states

class preblock(nn.Module):
    def __init__(self, input_chans, filter_size,num_features,pooling_size):
        super(preblock, self).__init__()
        self.filter_size=filter_size
        if type(self.filter_size)==tuple:  # filter_size may not be a square 3D input
            self.padding = ((self.filter_size[0] - 1) / 2, (self.filter_size[1] - 1) / 2, (self.filter_size[2] - 1) / 2)
        else:  # filter size = number
            self.padding = (self.filter_size - 1) / 2
        self.preblock=nn.Sequential(
        nn.Conv3d(input_chans, num_features, kernel_size=self.filter_size, stride=1, padding=self.padding),
        nn.BatchNorm3d(num_features), ###self.LN=nn.LayerNorm(24) not mentioned in CT1,but in ARS it's BN,
        nn.ELU(inplace=True), ##not important
        nn.AvgPool3d(kernel_size=pooling_size, stride=pooling_size)) # indicated in CT1 (1,1,2)


    def forward(self, x):
        x=self.preblock(x)
        return x

class ResCLSTM(nn.Module):
    def __init__(self, shape, input_chans, filter_size1,filter_size2,num_features,batch_size,scale_factor,num_layers=2):
        super(ResCLSTM, self).__init__()
        self.filter_size1 = filter_size1
        self.filter_size2 = filter_size2
        if type(self.filter_size2)==tuple:  # filter_size may not be a square
            self.padding2 = ((self.filter_size2[0] - 1) / 2, (self.filter_size2[1] - 1) / 2,(self.filter_size2[2] - 1) / 2)
        else:  # filter size = number
            self.padding2 = (self.filter_size2 - 2) / 2 #no need to be a tuple

        self.convlstm = CLSTM(shape=shape,input_chans=input_chans, filter_size=self.filter_size1,
                                  num_features=num_features,num_layers=num_layers,batch_size=batch_size,padding_mode='same',input_form='BCSWH')
        # Ref to Carl Thome's CLSTM code, neither SELU nor orthogonal initialization
        self.convlstm.apply(weights_init) # Initialize the weight
        self.hidden_state = self.convlstm.init_hidden()
        self.upsample = nn.interpolate(scale_factor=scale_factor,mode='linear',align_corners=True)
        #upsample is deprecated in favor of interpolate, but linear only supports the 3D input!!!!
        #self.upsample=nn.Upsample(scale_factor=scale_factor,mode='linear',align_corners=True)# over time CT1
        # if align_corners=True, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels
        self.conv=nn.Conv3d(num_features,num_features,self.filter_size2,stride=scale_factor,padding=self.padding) #???????
        #CT1, no explicite info to deal with the upsampled map, stride=2? filter_size=2*2????????

    def forward(self, x):
        residual = x
        _,x=self.convlstm(x)
        x=self.upsample(x)
        x=self.conv(x)
        x += residual
        x = self.relu(x) ###?no mentioned
        return x

class NiN(nn.Module): #LSTM
    def __init__(self, shape, input_chans,batch_size, filter_size,num_features1,num_features2,num_features3,last_AF=True):
        super(NiN, self).__init__() #(L + C (1×1) + B(LN?) + R) × 2 + L)
        self.last_AF=last_AF
        self.convlstm = CLSTM(shape,input_chans=num_features1, filter_size=filter_size,
                                  num_features=num_features2,num_layers=1,batch_size=batch_size,padding_mode='same',input_form='BCSWH') ####bad order,how to embed
        self.convlstm.apply(weights_init) # Initialize the weight
        self.hidden_state = self.convlstm.init_hidden()
        self.conv = nn.Conv3d(num_features2, num_features3, kernel_size=1) ##1*1*1 kernel,no need to define a kernel size
        if self.last_AF:
            self.ELU=nn.ELU(inplace=True)

        #其他层要不要定义？？？？？？？之前Conv中还加了dropout3d,而且可能还不work大小需要调整
    def forward(self, x):
        _,x=self.convlstm(x) #a special form convlstm=fully connected lstm,no need to write a new one and add a reshape
        x=self.conv(x)
        if self.last_AF:
            x=self.ELU(x)
        return x
    
    
class LSTM_cell(nn.Module):
    """Initialize a basic LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """
    def __init__(self, shape, input_chans, filter_size, num_features,batch_size,Layer_Normalization=True,peephole=True):
        super(LSTM_cell, self).__init__()

        self.shape = shape  # H,W
        self.filter_size = filter_size
        self.num_features = num_features
        self.batch_size=batch_size
        self.input_chans = input_chans
        self.Layer_normalization = Layer_Normalization
        self.peephole=peephole

        if self.Layer_normalization:
            self.LN=nn.LayerNorm([self.num_features,self.shape[0],self.shape[1]]) # no matter how many layers of LSTM
        self.linear = nn.Linear(self.input_chans + self.num_features,4 * self.num_features)

        if self.peephole:
            self.wi = Parameter(torch.Tensor(self.num_features, self.shape[0], self.shape[1]))#  Fully-Connected layer in paper is as same as CNN2d without flattening and extending
            self.wf = Parameter(torch.Tensor(self.num_features, self.shape[0], self.shape[1]))
            self.wo = Parameter(torch.Tensor(self.num_features, self.shape[0], self.shape[1]))
            self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./ math.sqrt(sum(self.wi.size())) # what's the initializer?
        self.wi.data.uniform_(-stdv, stdv)
        self.wf.data.uniform_(-stdv, stdv)
        self.wo.data.uniform_(-stdv, stdv)


    def forward(self, input, hidden_state):
        hidden, c = hidden_state  # hidden and c are regarded as images with several channels
        combined = torch.cat((input, hidden), 1)  # concatenate in the channels
        combined = combined.transpose(1,3)
        A = self.linear(combined)
        A = A.transpose(1,3)
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)  # it should return 4 tensors, equally split
        if self.peephole:
            ai+=self.wi*c
            af+=self.wf*c

        if self.Layer_normalization:
            af = self.LN(af)
            ag = self.LN(ag)
            ai = self.LN(ai)

        i = torch.sigmoid(ai)
        f = torch.sigmoid(af+1.0) # WARNING:After layer normalization, the forget gate bias is eliminated, and its value is close to 0.5, +1 if LN
        g = torch.tanh(ag)

        next_c = f * c + i * g

        if self.peephole:
            ao += self.wo*next_c

        if self.Layer_normalization:
            ao=self.LN(ao)
            next_c=self.LN(next_c)

        o=torch.sigmoid(ao)
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self):
        return (torch.zeros(self.batch_size, self.num_features, self.shape[0], self.shape[1]).to(dev),  #h0
                torch.zeros(self.batch_size, self.num_features, self.shape[0], self.shape[1]).to(dev))  #c0  same size initialized by zeros




class LSTM(nn.Module):
    """Initialize a basic  LSTM cell.
       for a whole unfold LSTM(with len)
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """
    def __init__(self, shape, input_chans, filter_size, num_features, num_layers,batch_size,Layer_Normalization=True,peephole=True):
        super(LSTM, self).__init__()
        self.shape = shape  # H,W
        self.filter_size = filter_size
        self.num_features = num_features
        self.num_layers = num_layers
        self.input_chans=input_chans
        self.batch_size=batch_size
        self.Layer_Normalization=Layer_Normalization
        self.peephole=peephole
        cell_list = []
        cell_list.append(
            LSTM_cell(self.shape, input_chans, self.filter_size, self.num_features,self.batch_size,self.Layer_Normalization,
                       self.peephole).to(dev))  # the first
        # one has a different number of input channels

        for idcell in xrange(1, self.num_layers):
            cell_list.append(CLSTM_cell(self.shape, input_chans, self.num_features, self.filter_size, self.num_features,self.batch_size,
                                        self.Layer_Normalization,self.peephole).to(dev))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, current_input, hidden_state):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape seq_len,Batch,Chans,H,W

        """      
        seq_len = current_input.size(1)
        next_hidden = []  # hidden states(h and c)

        for idlayer in xrange(self.num_layers):  # loop for every layer

            hidden_c = hidden_state[idlayer]  # hidden and c are images with several channels
            all_output = []
            output_inner = []
            for t in xrange(seq_len):  # loop for every step
                hidden_c = self.cell_list[idlayer](current_input[:,t, ...],
                                                   hidden_c)  # cell_list is a list with different conv_lstms 1 for every layer  
                output_inner.append(hidden_c[0]) #hidden_c[0]=hidden,hidden_c[1]=c
            next_hidden.append(hidden_c) #the last hidden c output
            current_input = torch.cat(output_inner, 1).view(output_inner[0].size(0),seq_len,
                                                            *output_inner[0].size()[1:])  #*output_inner[0].size() before
        return next_hidden, current_input  # get back to the shape: B,seq_len,chans,H,W

    def init_hidden(self): # no relation with layer numbers
        init_states = []  # this is a list of tuples
        for i in xrange(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden())
        return init_states    
    
    