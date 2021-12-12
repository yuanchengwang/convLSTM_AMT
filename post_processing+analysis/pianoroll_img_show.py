#post_processing
#pianoroll_img
#   plot the image of label and corresponding prediction output
#       Blue:correct points
#       Red: missing points
#       Yellow: flase points
#       Green: onsets
import numpy as np
from pypianoroll import Multitrack, Track
from matplotlib import pyplot as plt
##################paths###################
path_label = '/home/wyc/Desktop/MAPS_MUS-alb_se2_ENSTDkCl_label.npz'
path_prediction = '/home/wyc/Desktop/prediction.npz'
path_save = '/home/wyc/Desktop/pianoroll_img.png'
################parameter#################
RangeMIDInotes = [21,108]

def pianoroll_img(path_label,path_prediction,path_save,RangeMIDInotes,length = None,Mix = False): 
    #length:showing length,format:[start_index,end_index],default:None
    #Mix:(False,show the label and prediction separatly),(True,show the result of mix result)
    #       Blue:correct points
    #       Red: missing points
    #       Yellow: flase points
    #       Green: onsets
#    assert(not(length == None  and  Mix == True))
    prediction = np.load(path_prediction)['arr_0']
    label = np.load(path_label)['arr_0']
    uppadding = 128-max(RangeMIDInotes)
    downpadding = min(RangeMIDInotes)-1
    if length == None:
        prediction = np.concatenate( [np.zeros((prediction.shape[0],20)),prediction,np.zeros((prediction.shape[0],20))],1)
        label = np.concatenate( [np.zeros((label.shape[0],20)),label,np.zeros((label.shape[0],20))],1)
    else:
        prediction = np.concatenate( [np.zeros((length[1]-length[0],downpadding)),prediction[length[0]:length[1]],np.zeros((length[1]-length[0],uppadding))],1)
        label = np.concatenate( [np.zeros((length[1]-length[0],downpadding)),label[length[0]:length[1]],np.zeros((length[1]-length[0],uppadding))],1)
    print(prediction.shape)
    print(label.shape)
    if Mix == False:
        track_prediction = Track(pianoroll=prediction, program=0, is_drum=False,
              name='prediction')
        track_label = Track(pianoroll=label, program=0, is_drum=False,
              name='label')
        multitrack = Multitrack(tracks=[track_prediction, track_label])
        fig,ax = multitrack.plot()
        plt.savefig(path_save)
    else:
        True_point = np.ones_like(label)
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):                
                if label[i,j] == 1 and label[i-1,j] == 0:
                    True_point[i,j] = 100.0/256  #green,onset
                elif label[i,j] == 1 and prediction[i,j] == 1:
                    True_point[i,j] = 20.0/256 #blue,true point
                elif label[i,j] == 1 and prediction[i,j] == 0:
                    True_point[i,j] = 190.0/256 #red,miss 
                elif label[i,j] == 0 and prediction[i,j] == 1:
                    True_point[i,j] = 140.0/256 #yellow,false                
        track = Track(pianoroll = True_point,program=0, is_drum=False,
              name='label')
        fig, ax = track.plot(cmap = 'gist_ncar')
        plt.savefig(path_save)      
        
        
        
        
        
pianoroll_img(path_label,path_prediction,path_save,RangeMIDInotes,length = [0,12000],Mix = True)



