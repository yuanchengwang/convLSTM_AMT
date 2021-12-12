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
#import warnings
#warnings.filterwarnings('ignore') # need to filter the connexion issue???

eps=sys.float_info.epsilon
dev=torch.device('cuda') # it works
#dev=torch.device('cuda:0')
viz = visdom.Visdom()
#assert viz.check_connection()


########## PATH ############
#data_dir='/home/wyc/Desktop/preprocessed_data'
#data_dir='/home/wyc/Desktop/toy_dataset'
#data_dir='/home/wyc/Desktop/toy_dataset_dual'
#data_dir='/home/wyc/Desktop/preprocessed_data_dual'
data_dir='/home/wyc/Desktop/preprocessed_data_onset'
save_dir='/home/wyc/Desktop/model_save'  #model save
if not os.path.exists(save_dir):
        os.makedirs(save_dir)

############ Training parameters ###########
nb_epochs=config['epoch']
n_workers=4  #data loader workers, previous value=5 #n_gpu=3
start_lr=0.01
weight_decay=1e-4
save_freq=1

kernel_size=config['kernel_size']
win_width=config['win_width']  #label window, input window=win_width+kernel_size-1,in sequence, =seq_len
batch_size=config['batch_size']  #256=32*8=256*1
model_path=config['model']  #choose different model: 'conv','convlstm1','convlstm2'
dual_train=config['dual_train']
input_chans=config['input_chans']

line = viz.line(np.arange(10))  #line plot visualization for training progress
text = viz.text("<h1>{} model</h1>".format(model_path))

model = import_module(model_path)
net, loss= model.get_model()
# n_gpu = setgpu(n_gpu)
# args.n_gpu = n_gpu
# if model_path=='convlstm':
#     net.apply()  # weight_intialization is performed in net init
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

# dataset=data_loader.data_loader(data_dir,win_width, kernel_size,overlap=True,phase='test',dual_train=dual_train)
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


def get_lr(epoch,nb_epochs,start_lr):
    if epoch <= nb_epochs * 0.5:
        lr = start_lr
    elif epoch <= nb_epochs * 0.8:
        lr = 0.1 * start_lr
    else:
        lr = 0.01 * start_lr
    return lr


def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir,nb_epochs,start_lr,start_time,time_p,loss_p,Fscore,dual_train):
    start_time1 = time.time()
    assert (dual_train == 'Frame' or dual_train == 'Onset' or  dual_train == 'Both')
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
        loss_output = loss(output,target,dual_train=dual_train) #(8L, 88L, 32L, 1L)/(8L, 1L, 32L, 88L)
        optimizer.zero_grad()        
        loss_output[0].backward()        
        optimizer.step()
        
        if dual_train == 'Frame':
            loss_output[0] = loss_output[0].item()
            metrics_frame.append(loss_output[:])
            metrics_temp.append(loss_output[:])
        elif dual_train == 'Onset':
            metrics_loss.append(loss_output[0].item())
            metrics_onset.append(loss_output[1])
        else: 
            metrics_loss.append(loss_output[1])
            metrics_frame.append(loss_output[2])
            metrics_onset.append(loss_output[3]) 
#        if i % 1000000==0:#% 1000000==0: #same as once per epoch, only for loss, the framewise evaluation must be calculated within a whole epoch
#            if i !=0:
#                metrics_temp=np.asarray(metrics_temp, np.float32)
#                loss_p.append(np.mean(metrics_temp[:,0]))
#                TP = np.sum(metrics_temp[:, 1])
#                Precision = TP / (np.sum(metrics_temp[:, 2]) + eps)
#                Recall = TP / (np.sum(metrics_temp[:, 3]) + eps)
#            else:
#                loss_p.append(metrics_temp[0][0])
#                TP = metrics_temp[0][1]
#                Precision = TP / (metrics_temp[0][2] + eps)
#                Recall = TP / (metrics_temp[0][3] + eps)
#            Fscore.append(2 * Precision * Recall / (Precision + Recall + eps))
#            time_p.append(time.time() - start_time)
#            #print np.array(Fscore),np.array(loss_p),np.array(time_p)
#            viz.line(X=np.column_stack((np.array(time_p), np.array(time_p))),
#                     Y=np.column_stack((np.array(loss_p), np.array(Fscore))),
#                     win=line, opts=dict(legend=["Loss", "TRAIN_Fscore"]))
#            # only print in visdom text window
#            viz.text("<p style='color:red'>epoch:{}</p><br><p style='color:blue'>Loss:{:.4f}</p><br>"
#                     "<p style='color:BlueViolet'>TRAIN_Fscore:{:.4f}</p><br><p style='color:green'>Time:{:.2f}</p>".format(epoch, loss_p[-1], Fscore[-1], time.time()-start_time), win=text)
#            # <p style='color:orange'>TEST_acc:{:.4f}</p><br>"
#            metrics_temp=[]      
    
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
        
    if dual_train == 'Frame':
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
        
        if dual_train == 'Both':         
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
        if dual_train == 'Both': 
            print('Train_frame: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f' % (np.mean(metrics_loss[:][0]),Precision_frame,Recall_frame,Fscore_epoch_frame))
            print('Train_onset: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f' % (np.mean(metrics_loss[:][1]),Precision_onset,Recall_onset,Fscore_epoch_onset))
        else:
            print('Train_onset: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f' % (np.mean(metrics_loss[:]),Precision_onset,Recall_onset,Fscore_epoch_onset))
        print
        with open(save_dir+"/log.txt","a") as f:
            f.write('Epoch %03d (lr %.5f),time %3.2f\n' % (epoch, lr,time.time() - start_time1))
            if dual_train == 'Both': 
                f.write('Train_frame: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f\n' % (np.mean(metrics_loss[:][0]),Precision_frame,Recall_frame,Fscore_epoch_frame))
                f.write('Train_onset: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f\n' % (np.mean(metrics_loss[:][1]),Precision_onset,Recall_onset,Fscore_epoch_onset))
            else:
                f.write('Train_onset: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f\n' % (np.mean(metrics_loss[:]),Precision_onset,Recall_onset,Fscore_epoch_onset))
    return time_p,loss_p,Fscore

def validate(data_loader, net, loss , dual_train,save_dir):
    start_time = time.time()

    net.eval()

    metrics_frame = []
    metrics_loss,metrics_onset = [],[]  
    
    with torch.no_grad(): 
        for i, (data, target) in enumerate(data_loader):
            data = data.to(dev)
            target = target.to(dev)

            output = net(data)
            loss_output = loss(output, target, dual_train=dual_train)            
            
            if dual_train == 'Frame':
                loss_output[0] = loss_output[0].item()
                metrics_frame.append(loss_output[:])
            elif dual_train == 'Onset':
                metrics_loss.append(loss_output[0].item())
                metrics_onset.append(loss_output[1])
            else:
                metrics_loss.append(loss_output[1])
                metrics_frame.append(loss_output[2]) 
                metrics_onset.append(loss_output[3])
                
    end_time = time.time()
    
    if dual_train == 'Frame':
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
        if dual_train == 'Both':  
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
        if dual_train == 'Both': 
            print('Validation_frame: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f' % (np.mean(metrics_loss[:][0]),Precision_frame,Recall_frame,Fscore_epoch_frame))
            print('Validation_onset: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f' % (np.mean(metrics_loss[:][1]),Precision_onset,Recall_onset,Fscore_epoch_onset))
        else:
            print('Validation_onset: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f' % (np.mean(metrics_loss[:]),Precision_onset,Recall_onset,Fscore_epoch_onset))
        print
        with open(save_dir+"/log.txt","a") as f:
            f.write('Epoch %03d ,time %3.2f\n' % (epoch, end_time - start_time))
            if dual_train == 'Both': 
                f.write('Validation_frame: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f\n' % (np.mean(metrics_loss[:][0]),Precision_frame,Recall_frame,Fscore_epoch_frame))
                f.write('Validation_onset: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f\n' % (np.mean(metrics_loss[:][1]),Precision_onset,Recall_onset,Fscore_epoch_onset))
            else:
                f.write('Validation_onset: loss %2.4f, Framewise Precision %3.2f,Recall %3.2f, F-score %3.2f\n' % (np.mean(metrics_loss[:]),Precision_onset,Recall_onset,Fscore_epoch_onset))


if __name__=='__main__':
    start_epoch=0
    start_time=time.time()
    time_p,loss_p,Fscore=[],[],[]
    for epoch in range(start_epoch,nb_epochs):
        #print time_p,loss_p,Fscore
        time_p,loss_p,Fscore=train(train_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir,nb_epochs,start_lr,start_time,time_p,loss_p,Fscore,dual_train)
        validate(val_loader, net, loss,dual_train,save_dir)

# onset tracking evaluation
