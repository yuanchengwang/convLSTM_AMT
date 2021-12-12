import Post
import numpy as np
#############################
min_midi_pitch = 20
frame_length_seconds = 11.6   #ms
min_duration_ms = 46.4 # ms
n_note = 88
#############################
label_dir = '/home/wyc/Desktop/test_save/MAPS_MUS-alb_se2_ENSTDkCl_label.npz'
prediction_dir = '/home/wyc/Desktop/test_save/MAPS_MUS-alb_se2_ENSTDkCl_prediction.npz'
path_save = '/home/wyc/Desktop/test_save'


if __name__=='__main__': #test file
    label = np.load(label_dir)['arr_0']    
    label = label.transpose(2,1,0).reshape(label.shape[2],label.shape[1])

    label = label[:88,:20224] #####
    print label.shape
    label_note = label[:n_note,:].transpose(1,0)
#    label_onset = label[n_note:,:].transpose(1,0)
    
    prediction = np.load(prediction_dir)['arr_0']
    print prediction.shape
    prediction_note = prediction[:n_note,:].transpose(1,0)
#    prediction_onset = prediction[n_note:,:].transpose(1,0)

    Framewise = Post.frame_measure(prediction_note,label_note)
    print Framewise 
    
    label_interval,label_pitch = Post.pianoroll_to_note(metrics_note = label_note,metrics_onset = None,
    							min_midi_pitch = min_midi_pitch,frame_length_seconds = frame_length_seconds,min_duration_ms = min_duration_ms)
    prediction_interval,prediction_pitch = Post.pianoroll_to_note(metrics_note = prediction_note,metrics_onset = None,
    							min_midi_pitch = min_midi_pitch,frame_length_seconds = frame_length_seconds,min_duration_ms = min_duration_ms)
#    print prediction_interval[0:100]
    Notewise = Post.note_measure(label_interval,label_pitch,prediction_interval,prediction_pitch,False)
    print Notewise
    Notewise_offset = Post.note_measure(label_interval,label_pitch,prediction_interval,prediction_pitch,True)
    print Notewise_offset
#    #############################################################
    label_list = Post.counter(data = label_note,n_note = n_note,Type = 'frame')
    true_list =  Post.counter(data = label_note*prediction_note,n_note = n_note,Type = 'frame')
    Post.histogram_img(True_list = true_list,Label_list = label_list,measures = Framewise,
    					title = 'frame',path_save = path_save,Pitch_min = min_midi_pitch,name = 'frame_histogram')
    
#    Post.pianoroll_img(path_label = label_dir,path_prediction = prediction_dir,path_save = path_save,RangeMIDInotes = [20,88],name = 'untitled',length = None,Mix = True)
