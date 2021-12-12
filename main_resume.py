import numpy as np
import os
from importlib import import_module
import time

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
#from torch.autograd import Variable
import data_loader
import sys
from config import *
import visdom
os.environ['CUDA_VISIBLE_DEVICES']='0'
eps=sys.float_info.epsilon
dev=torch.device('cuda')
viz = visdom.Visdom()

########## PATH ############
#data_dir='/home/wyc/Desktop/preprocessed_data'
#data_dir='/home/wyc/Desktop/toy_dataset'
save_dir='/home/wyc/Desktop/model_save'  #model save
#data_dir='/home/wyc/Desktop/toy_dataset_dual'
data_dir='/home/wyc/Desktop/preprocessed_data_dual'
if not os.path.exists(save_dir):
        os.makedirs(save_dir)

############ Training parameters ###########
nb_epochs=config['epoch']
n_workers=4  #data loader workers, previous value=5 #n_gpu=3
start_lr=0.01
weight_decay=1e-4
save_freq=1

kernel_size=config['kernel_size']
win_width=config['win_width']  #label window, input window=win_width+kernel_size-1
batch_size=config['batch_size']  #256=32*8=256*1
model_path=config['model']  #choose different model: 'conv','convlstm1','convlstm2'
dual_train=config['dual_train']
input_chans=config['input_chans']
#model = import_module(model_path)
#net, loss= model.get_model()

######### Resume definition #############
start_epoch=6
resume_file='006.ckpt'
resume_path=os.path.join(save_dir,resume_file)
model = import_module(model_path)
net, loss= model.get_model()
checkpoint = torch.load(resume_path)
start_epoch = checkpoint['epoch'] + 1


net.load_state_dict(checkpoint['state_dict'])

net = net.to(dev)
loss = loss.to(dev)
cudnn.benchmark = True
net = DataParallel(net)  #neighter gpu set nor device_ids=[1,2,3] means using all GPUs

dataset=data_loader.data_loader(data_dir,win_width,input_chans, kernel_size,phase='train',model_path=model_path,dual_train=dual_train)
train_loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = n_workers,
        pin_memory=True,drop_last=True)   #train/val pin_memory=True, test pin_memory=False

dataset=data_loader.data_loader(data_dir,win_width,input_chans, kernel_size,phase='val',model_path=model_path,dual_train=dual_train)
val_loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = n_workers,
        pin_memory=True,drop_last=True)   #train/val pin_memory=True, test pin_memory=False

# dataset=data_loader.data_loader(data_dir,win_width, kernel_size,overlap=True,phase='test')
# test_loader = DataLoader(
#         dataset,
#         batch_size = batch_size,
#         shuffle = False,
#         num_workers = n_workers,
#         pin_memory=True)   #train/val pin_memory=True, test pin_memory=False,not the real test

optimizer = optim.SGD(
        net.parameters(),
        start_lr,
        momentum = 0.9,
        weight_decay = weight_decay)


def get_lr(epoch,nb_epochs,start_lr): ##### WARNING: Learning rate is defined as the same as the main, change it if need #####
    if epoch <= nb_epochs * 0.5:
        lr = start_lr
    elif epoch <= nb_epochs * 0.8:
        lr = 0.1 * start_lr
    else:
        lr = 0.01 * start_lr
    return lr


def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir,nb_epochs,start_lr,dual_train):
    start_time1 = time.time()

    net.train()
    lr = get_lr(epoch,nb_epochs,start_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics_frame,metrics_temp = [],[]
    metrics_loss,metrics_onset = [],[]

    for i, (data, target) in enumerate(data_loader): #iteration
        data = data.to(dev) #(16,32, 1, 7, 252)
        target = target.to(dev)
        output = net(data)
        loss_output = loss(output,target,onset=dual_train) #(8L, 88L, 32L, 1L)/(8L, 1L, 32L, 88L)
        optimizer.zero_grad()        
        loss_output[0].backward()        
        optimizer.step()
        
        if not dual_train:
            loss_output[0] = loss_output[0].item()
            metrics_frame.append(loss_output[:])
            metrics_temp.append(loss_output[:])
        else:
            metrics_loss.append(loss_output[1])
            metrics_frame.append(loss_output[2])
            metrics_onset.append(loss_output[3]) 
            
            

    if epoch % save_freq == 0:
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict
            },
            os.path.join(save_dir, '%03d.ckpt' % epoch))

    if not dual_train:
        metrics_frame = np.asarray(metrics_frame, np.float32)     
        TP=np.sum(metrics_frame[:, 1][0])
        Precision=TP/(np.sum(metrics_frame[:, 2][0])+eps)
        Recall=TP/(np.sum(metrics_frame[:, 3][0])+eps)
        Fscore_epoch=2*Precision*Recall/(Precision+Recall+eps)
        
        print('Epoch %03d (lr %.5f),time %3.2f' % (epoch, lr,time.time() - start_time1))  # Framewise and notewise Accuracy precision,recall,F-score 
        print('Train: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f' % (np.mean(metrics_frame[:,0]),Precision,Recall,Fscore_epoch))
        print
        with open(save_dir+"/log.txt","a") as f:
            f.write('Epoch %03d (lr %.5f),time %3.2f\n' % (epoch, lr,time.time() - start_time1))
            f.write('Train: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f\n' % (np.mean(metrics_frame[:,0]),Precision,Recall,Fscore_epoch))
    else:        
        metrics_frame = np.asarray(metrics_frame, np.float32)
        TP_frame = np.sum(metrics_frame[:,0])
        Precision_frame = TP_frame/(np.sum(metrics_frame[:,1])+eps)
        Recall_frame = TP_frame/(np.sum(metrics_frame[:,2])+eps)
        Fscore_epoch_frame = 2*Precision_frame*Recall_frame/(Precision_frame+Recall_frame+eps)
        
        metrics_onset = np.asarray(metrics_onset, np.float32)
        TP_onset = np.sum(metrics_onset[:,0])
        Precision_onset = TP_onset/(np.sum(metrics_onset[:,1])+eps)
        Recall_onset = TP_onset/(np.sum(metrics_onset[:,2])+eps)
        Fscore_epoch_onset = 2*Precision_onset*Recall_onset/(Precision_onset+Recall_onset+eps)
        
        print('Epoch %03d (lr %.5f),time %3.2f' % (epoch, lr,time.time() - start_time1))  # Framewise and notewise Accuracy precision,recall,F-score 
        print('Train_frame: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f' % (np.mean(metrics_loss[:][0]),Precision_frame,Recall_frame,Fscore_epoch_frame))
        print('Train_onset: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f' % (np.mean(metrics_loss[:][1]),Precision_onset,Recall_onset,Fscore_epoch_onset))
        print
        with open(save_dir+"/log.txt","a") as f:
            f.write('Epoch %03d (lr %.5f),time %3.2f\n' % (epoch, lr,time.time() - start_time1))
            f.write('Train_frame: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f\n' % (np.mean(metrics_loss[:][0]),Precision_frame,Recall_frame,Fscore_epoch_frame))
            f.write('Train_onset: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f\n' % (np.mean(metrics_loss[:][1]),Precision_onset,Recall_onset,Fscore_epoch_onset))

def validate(data_loader, net, loss,dual_train,save_dir):
    start_time = time.time()

    net.eval()

    metrics_frame = []
    metrics_loss,metrics_onset = [],[]  
    
    with torch.no_grad(): 
        for i, (data, target) in enumerate(data_loader):
            data = data.to(dev)
            target = target.to(dev)

            output = net(data)
            loss_output = loss(output, target, onset=dual_train)            
            
            if not dual_train:
                loss_output[0] = loss_output[0].item()
                metrics_frame.append(loss_output[:])
            else:
                metrics_loss.append(loss_output[1])
                metrics_frame.append(loss_output[2]) 
                metrics_onset.append(loss_output[3])

    end_time = time.time()
    if not dual_train:
        metrics_frame = np.asarray(metrics_frame, np.float32)
        TP=np.sum(metrics_frame[:, 1])
        Precision=TP/(np.sum(metrics_frame[:, 2])+eps)
        Recall=TP/(np.sum(metrics_frame[:, 3])+eps)
        Fscore=2*Precision*Recall/(Precision+Recall+eps)
        
        print('Epoch %03d ,time %3.2f' % (epoch, end_time - start_time))
        print('Validation: Loss %2.4f,Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f' % (np.mean(metrics_frame[:,0]),Precision,Recall,Fscore))
        print
        with open(save_dir+"/log.txt","a") as f:
            f.write('Epoch %03d ,time %3.2f\n' % (epoch, end_time - start_time))
            f.write('Validation: Loss %2.4f,Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f\n' % (np.mean(metrics_frame[:,0]),Precision,Recall,Fscore))
    else:
        metrics_frame = np.asarray(metrics_frame, np.float32)
        TP_frame = np.sum(metrics_frame[:,0])
        Precision_frame = TP_frame/(np.sum(metrics_frame[:,1])+eps)
        Recall_frame = TP_frame/(np.sum(metrics_frame[:,2])+eps)
        Fscore_epoch_frame = 2*Precision_frame*Recall_frame/(Precision_frame+Recall_frame+eps)
        
        metrics_onset = np.asarray(metrics_onset, np.float32)
        TP_onset = np.sum(metrics_onset[:,0])
        Precision_onset = TP_onset/(np.sum(metrics_onset[:,1])+eps)
        Recall_onset = TP_onset/(np.sum(metrics_onset[:,2])+eps)
        Fscore_epoch_onset = 2*Precision_onset*Recall_onset/(Precision_onset+Recall_onset+eps)
        
        print('Epoch %03d ,time %3.2f' % (epoch, end_time - start_time))  # Framewise and notewise Accuracy precision,recall,F-score 
        print('Validation_frame: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f' % (np.mean(metrics_loss[:][0]),Precision_frame,Recall_frame,Fscore_epoch_frame))
        print('Validation_onset: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f' % (np.mean(metrics_loss[:][1]),Precision_onset,Recall_onset,Fscore_epoch_onset))
        print
        with open(save_dir+"/log.txt","a") as f:
            f.write('Epoch %03d ,time %3.2f\n' % (epoch, end_time - start_time))
            f.write('Validation_frame: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f\n' % (np.mean(metrics_loss[:][0]),Precision_frame,Recall_frame,Fscore_epoch_frame))
            f.write('Validation_onset: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f\n' % (np.mean(metrics_loss[:][1]),Precision_onset,Recall_onset,Fscore_epoch_onset))


if __name__=='__main__':
    for epoch in range(start_epoch,nb_epochs):
        train(train_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir,nb_epochs,start_lr,dual_train)
        validate(val_loader, net, loss,dual_train,save_dir)

# onset tracking evaluation
