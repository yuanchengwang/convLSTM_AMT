import numpy as np
from mir_eval import transcription
import torch
import os
import sys
########post parameters#############
eps = sys.float_info.epsilon


def pianoroll_process(raw_matrix,dual_train = False):
	#transform the raw output&label matrix to typical pianoroll
    pianoroll = (torch.sigmoid(raw_matrix) >= 0.5).type(torch.cuda.FloatTensor)
    if dual_train:
        raw_matrix = raw_matrix.transpose(0,1).reshape(raw_matrix.shape[1],-1)
        pianoroll = pianoroll.transpose(0,1).reshape(pianoroll.shape[1],-1)

        pianoroll_frame = pianoroll[:pianoroll.shape[0]/2,:]
        pianoroll_onset = pianoroll[pianoroll.shape[0]/2:,:]
        

        
        pianoroll_diff = torch.zeros_like(pianoroll_onset)
        for i in range(pianoroll_diff.shape[1]):
            pianoroll_diff[:,i] = pianoroll_onset[:,i] - pianoroll_onset[:,i-1]
        pianoroll_diff_on = (pianoroll_diff > 0).type(torch.cuda.FloatTensor)
        onset_list = torch.nonzero(pianoroll_diff_on)
        
        pianoroll_onset = torch.zeros_like(pianoroll_onset)
        for onset in onset_list:
            	length = torch.argmax(pianoroll_diff_on[onset[0],onset[1]:onset[1]+5])   #####5??
        pianoroll_onset[onset[0],(onset[1]+length):(onset[1]+length+5)] = 1
        pianoroll = torch.cat((pianoroll_frame,pianoroll_onset),dim = 0)
    return pianoroll

def pianoroll_to_note(metrics_note,metrics_onset,min_midi_pitch, frames_per_second, min_duration_ms,save_dir = None, name = None):
#	Reference:ONSETS AND FRAMES: DUAL-OBJECTIVE PIANO TRANSCRIPTION
    frame_length_seconds = 1.0/frames_per_second
    pitch_start_step = {}
#    pitch_end_step = {} 
    sequence_interval = []
    sequence_pitch = []

    metrics_note = np.append(metrics_note, [np.zeros(metrics_note[0].shape)], 0)
    if metrics_onset is not None:
        metrics_onset = np.append(metrics_onset, [np.zeros(metrics_onset[0].shape)], 0)
    # Ensure that any frame with an onset prediction is considered active.
        frames = np.logical_or(metrics_note, metrics_onset)

    def process_active_pitch(pitch, i):
#    Process a pitch being active in a given frame.
        if pitch not in pitch_start_step:
            if metrics_onset is not None:
                # If onset predictions were supplied, only allow a new note to start
                # if we've predicted an onset.
                if metrics_onset[i, pitch]:
                    pitch_start_step[pitch] = i          
                else:
                    # Even though the frame is active, the onset predictor doesn't
                    # say there should be an onset, so ignore it. 
                    pass        
            else:                
                pitch_start_step[pitch] = i
        else:
            if metrics_onset is not None:
            # pitch is already active, but if this is a new onset, we should end
            # the note and start a new one.
                if (metrics_onset[i, pitch] and
                    not metrics_onset[i - 1, pitch]):
                    end_pitch(pitch, i)
                    pitch_start_step[pitch] = i

    def end_pitch(pitch, end_frame):
#        End an active pitch.
        start_time = pitch_start_step[pitch] * frame_length_seconds
        end_time = end_frame * frame_length_seconds

        if (end_time - start_time) * 1000 >= min_duration_ms:
            sequence_interval.append([start_time,end_time])
            sequence_pitch.append(pitch + min_midi_pitch)
            # pitch_end_step[pitch] = [pitch_start_step[pitch],end_frame]
        del pitch_start_step[pitch]

    for i, frame in enumerate(frames):
        for pitch, active in enumerate(frame):
            if active:
                process_active_pitch(pitch, i)
            elif pitch in pitch_start_step:
                end_pitch(pitch, i)

    return sequence_interval,sequence_pitch


#def note_measure(sequence_target, pitch_target, sequence_output, pitch_output,with_offset = False):
#	if with_offset:
#        precision,recall,F1,overlap = transcription.precision_recall_f1_overlap(sequence_target, pitch_target, sequence_output, pitch_output)
#    else:
#        precision,recall,F1,overlap = transcription.precision_recall_f1_overlap(sequence_target, pitch_target, sequence_output, pitch_output,offset_ratio = None)
#    
#	return precision,recall,F1

def frame_measure(prediction,label):

	Pos_TP = prediction*label
	TP = Pos_TP.sum()

	precision = TP/(prediction.sum()+eps)
	recall = TP/(label.sum()+eps)
	F1 = 2*precision*recall/(precision+recall+eps)

	return precision,recall,F1

def counter(data,n_note,Type): # calculte the number of note&frame per pitch
	assert(Type == 'frame' or Type == 'note')
	if Type == 'frame':
		note_list = torch.sum(data,1)
	else:
		note_list = torch.zeros(n_note,1) 
		for note in data:
			note_list[note] = note_list[note] + 1

	return note_list


#def histogram_img(True_list, Label_list,measures, title = 'untitled',path_save = None,Pitch_min = 1,name = 'untitled'): 
#    #True_list:number of true notes&frames per pitch
#    #Label_list:number of Label notes&frames per pitch
#    #measures:frame&notebase P/R/F
#    #title: histogram title as you wish
#    #path_save: png saving path
#    #Note_min: the midi index of lowest pitch
#    #name: file name as you wish
#    plt.figure(figsize=(9,6))     
#    X = np.arange(len(True_list))+Pitch_min
#    Texty = np.ceil(max(Label_list)*15.0/16)
#    Textx = np.ceil(len(True_list)*13.0/16 + Pitch_min)     
#    plt.bar(X,Label_list,width = 1,facecolor = 'plum',edgecolor = 'white',label = (('Total')))
#    plt.bar(X,True_list,width = 1,facecolor = 'lightskyblue',edgecolor = 'white',label = (('Correct')))
#    plt.legend(loc = 'upper left')
#    plt.annotate('PRESISION:%s\nRECALL:%s\nFMEASURE:%s'% (measures[0],measures[1],measures[2]),xy = (Textx,Texty),
#                 bbox = dict(boxstyle="square,pad=0.3",fc="w"),xytext = (Textx,Texty),size = 10,va = 'center',ha = 'center')
#    plt.title(title)
#    if path_save != None:
#        if not os.path.exists(path_save):
#            os.mkdir(path_save)
#        path = os.path.join(path_save,name)
#        plt.savefig(path)
#
#
#def pianoroll_img(path_label,path_prediction,path_save = None,RangeMIDInotes = [1,128],name = 'untitled',length = None,Mix = False): 
#    #length:showing length,format:[start_index,end_index],default:None
#    #Mix:(False,show the label and prediction separatly),(True,show the result of mix result)
#    
##    assert(not(length == None  and  Mix == True))   #if the label and prediction length in same length
#    prediction = np.load(path_prediction)['arr_0']
#    label = np.load(path_label)['arr_0']
#    uppadding = 128-max(RangeMIDInotes)
#    downpadding = min(RangeMIDInotes)-1
#    if length == None:
#        prediction = np.concatenate( [np.zeros((prediction.shape[0],20)),prediction,np.zeros((prediction.shape[0],20))],1)
#        label = np.concatenate( [np.zeros((label.shape[0],20)),label,np.zeros((label.shape[0],20))],1)
#    else:
#        prediction = np.concatenate( [np.zeros((length[1]-length[0],downpadding)),prediction[length[0]:length[1]],np.zeros((length[1]-length[0],uppadding))],1)
#        label = np.concatenate( [np.zeros((length[1]-length[0],downpadding)),label[length[0]:length[1]],np.zeros((length[1]-length[0],uppadding))],1)
#    print(prediction.shape)
#    print(label.shape)
#    if Mix == False:
#        track_prediction = Track(pianoroll=prediction, program=0, is_drum=False,
#              name='prediction')
#        track_label = Track(pianoroll=label, program=0, is_drum=False,
#              name='label')
#        multitrack = Multitrack(tracks=[track_prediction, track_label])
#        fig,ax = multitrack.plot()
#        if path_save != None:
#            if not os.path.exists(path_save):
#                os.mkdir(path_save)
#            path = os.path.join(path_save,name)
#            plt.savefig(path)
#    else:
#        True_point = np.ones_like(label)
#        for i in range(label.shape[0]):
#            for j in range(label.shape[1]):                
#                if label[i,j] == 1 and label[i-1,j] == 0:
#                    True_point[i,j] = 100.0/256  #green,onset
#                elif label[i,j] == 1 and prediction[i,j] == 1:
#                    True_point[i,j] = 20.0/256 #blue,true point
#                elif label[i,j] == 1 and prediction[i,j] == 0:
#                    True_point[i,j] = 190.0/256 #red,miss 
#                elif label[i,j] == 0 and prediction[i,j] == 1:
#                    True_point[i,j] = 140.0/256 #yellow,false                
#        track = Track(pianoroll = True_point,program=0, is_drum=False,
#              name='mix')
#        fig, ax = track.plot(cmap = 'gist_ncar')
#        if path_save != None:
#            if not os.path.exists(path_save):
#                os.mkdir(path_save)
#            path = os.path.join(path_save,name)
#            plt.savefig(path)