import numpy as np
import os
from importlib import import_module
import time

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader

import Post
import data_loader
import sys
from config import *
eps=sys.float_info.epsilon
dev_cuda=torch.device('cuda') # it works
dev_cpu=torch.device('cpu')
########## PATH ############
#data_dir='/home/wyc/Desktop/preprocessed_data'
data_dir='/home/wyc/Desktop/toy_dataset_dual'
#model_dir='/home/wyc/Desktop/model_save_conv_valid_dual_train_80_80_nodropout'  #model save
model_dir='/home/wyc/Desktop/model_save_convlstm3_hcqt_only'  #model save
save_dir= '/home/wyc/Desktop/test_save/MAPS_MUS-alb_se2_ENSTDkCl_prediction.npz'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
############ Testing parameters ###########
n_workers=1  #n_workers=1 in test mod only
kernel_size=config['kernel_size']
win_width=config['win_width']  #label window, input window=win_width+kernel_size-1,in sequence, =seq_len
batch_size=config['batch_size']  #256=32*8=256*1
model_path=config['model']  #choose different model: 'conv','convlstm1','convlstm2'
dual_train=config['dual_train']
input_chans=config['input_chans']
######### test definition #############
test_file='011.ckpt'
test_path=os.path.join(model_dir,test_file)
model = import_module(model_path)
net, loss= model.get_model()
checkpoint = torch.load(test_path)
net.load_state_dict(checkpoint['state_dict'])

net = net.to(dev_cuda)
loss = loss.to(dev_cuda)
cudnn.benchmark = True
net = DataParallel(net)  #neighter gpu set nor device_ids=[1,2,3] means using all GPUs
    
dataset = data_loader.data_loader(data_dir,win_width,input_chans, kernel_size,phase = 'test',model_path = model_path,drop_last=True)
test_loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = n_workers,
        pin_memory=True,drop_last=True)   #train/val pin_memory=True, test pin_memory=False,not the real test

    
def test(data_loader,net,dual_train,save_dir):
    start_time = time.time()

    net.eval()
    prediction_list = []
    with torch.no_grad(): 
        for i, data in enumerate(data_loader):
            data = data.to(dev_cuda) 
            output = net(data)
            prediction = Post.pianoroll_process(output,dual_train)  
            prediction = prediction.to(dev_cpu)
            prediction_list.append(prediction)

        prediction_list = np.concatenate(prediction_list,1)

    np.savez_compressed(save_dir, prediction_list) #########

    end_time = time.time()
    print('time %3.2f' % (end_time - start_time))
    print(os.path.join(save_dir))


if __name__=='__main__':    
    test(test_loader,net,dual_train,save_dir)

