import numpy as np
import os
from importlib import import_module
import time

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
#from torch.autograd import Variable
import data_loader
import sys
from config import *
eps=sys.float_info.epsilon
dev_cuda=torch.device('cuda') # it works
dev_cpu=torch.device('cpu')
########## PATH ############
#data_dir='/home/wyc/Desktop/preprocessed_data'
data_dir='/home/wyc/Desktop/toy_dataset'
save_dir='/home/wyc/Desktop/model_save_convlstm1_same_peephole'  #model save
if not os.path.exists(save_dir):
        os.makedirs(save_dir)
test_dir= '/home/wyc/Desktop'
############ Testing parameters ###########
n_workers=1  #data loader workers, previous value=5 #n_gpu=3
kernel_size=config['kernel_size']
win_width=config['win_width']  #label window, input window=win_width+kernel_size-1,in sequence, =seq_len
batch_size=config['batch_size']  #256=32*8=256*1
model_path=config['model']  #choose different model: 'conv','convlstm1','convlstm2'

######### test definition #############
test_file='011.ckpt'
test_path=os.path.join(save_dir,test_file)
model = import_module(model_path)
net, loss= model.get_model()
checkpoint = torch.load(test_path)
net.load_state_dict(checkpoint['state_dict'])

net = net.to(dev_cuda)
loss = loss.to(dev_cuda)
cudnn.benchmark = True
net = DataParallel(net)  #neighter gpu set nor device_ids=[1,2,3] means using all GPUs

dataset = data_loader.data_loader(data_dir,win_width, kernel_size,phase='test',model_path=model_path,drop_last=True)
test_loader = DataLoader(
         dataset,
         batch_size = batch_size,
         shuffle = False,
         num_workers = n_workers,
         pin_memory=True)   #train/val pin_memory=True, test pin_memory=False,not the real test

def test(data_loader,net):
    start_time = time.time()

    net.eval()
#    block=[]
    prediction_list = []
#    label_list = []
    with torch.no_grad(): 
        for i, data in enumerate(data_loader):
            data = data.to(dev_cuda) 
            output = net(data)
            pos = (torch.sigmoid(output) >= 0.5).type(torch.cuda.FloatTensor)
            pos = pos.to(dev_cpu)
            pos = np.asarray(pos)            
            for j in range(pos.shape[0]):
                prediction_list.append(pos[j,:,:,:])
#                label_list.append(target[j,:,:,:])
#            print prediction_list[0].shape
        prediction_list = np.concatenate(prediction_list,1)
#        label_list = np.concatenate(label_list,1)        
#        print prediction_list.shape
        prediction_list = prediction_list.transpose(2,1,0)
#        label_list = label_list.transpose(2,1,0)
#        print target.shape
#    block.append(pos)
#    print block[0].shape
#    np.savez_compressed(test_dir + '/_test.npz', np.concatenate(block))
    np.savez_compressed(test_dir + '/prediction.npz', prediction_list[0])
#    np.savez_compressed(test_dir + '/label.npz', label_list[0])
    end_time = time.time()
    print('time %3.2f' % (end_time - start_time))


if __name__=='__main__':

    test(test_loader,net)
# onset tracking evaluation
