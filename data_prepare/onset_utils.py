# This is the onset processing toolbox

import numpy as np
# import pretty_midi
# import os
# import glob

parameter=2




def onset_smooth(matrix,smooth_mode='fuzzy',parameter=parameter):
    # this code is used to smooth the onset point, as the onset misalignment[1] and a little delay for the onset [2]
    # [1] Schluter, Jan, and S. Bock. "Improved musical onset detection with Convolutional Neural Networks." (2014):6979-6983.
    # [2] Curtis Hawthorne,Erich Elsen et al. "Onsets and Frames: Dual-Objective Piano Transcription." International Society For Music Information Retrieval Conference 2018.
    # input matrix is calculated by onset diff function
    # fuzzy for [1],parameter=std of gaussian distribution, 1 by default?????
    # delay for [2],parameter= delayed frame(s) related to the onset points,2 by default
    # Poisson, our proposed method for onset smoothing,parameter is the lambda in distribution,2 by default
    shape=matrix.shape #[88,seq_len]
    smoothed_matrix=np.zeros(shape)
    #assert(smooth_mode=='fuzzy' or smooth_mode == 'delay' or smooth_mode=='poisson')
    if smooth_mode=='fuzzy':#gauss function smoothing
        kernel=gauss(range(-2,3),parameter) #
        for i in xrange(shape[0]):
            smoothed_matrix[i,:]=np.convolve(kernel,matrix[i,:],mode='same')
        smoothed_matrix[smoothed_matrix>1]=1 # for close
    elif smooth_mode=='delay':
        for i in xrange(shape[0]):
            for j in xrange(shape[1]):
                if matrix[i,j]==1:
                    if j<shape[1]-parameter:
                        smoothed_matrix[i,j+1:j+parameter+1]=1
                    else:
                        smoothed_matrix[i,j:]=1
    else: #Poisson
        kernel=poisson(range(0,4),parameter)
        kernel/=np.max(kernel)
        for i in xrange(shape[0]):
            smoothed_matrix[i,:]=np.convolve(kernel,matrix[i,:],mode='same')
        smoothed_matrix[smoothed_matrix>1]=1 # for interval [0,1]
    return smoothed_matrix

def onset_matrix(midi_data,fs,matrix_len,smooth=True,smooth_mode='fuzzy',parameter=parameter):
    onset=get_onsets_with_notes(midi_data.notes,fs,matrix_len) #get the 0-1 matrix while 1 for the note onset,[len,128]
    if smooth:
        onset=onset_smooth(onset,smooth_mode=smooth_mode,parameter=parameter)
    return onset

def comb_dim(label,axis=0):
    # for multi-channel combinaison
    assert(len(label.shape)==3)
    label_comb=np.sum(label,axis) #drop a dimension
    label_comb[label_comb>0]=1
    return np.expand_dims(label_comb,axis)

def gauss(x, sigma=1.): # Gaussian kernel(not distribution, 1/2pi*sigma^2 eliminated)
    return np.exp(-np.array(x).astype('float')**2/2/float(sigma)**2)#/np.sqrt(2*np.pi)/float(sigma)

def poisson(x,lamb=1.2): #lambda which is an in-python parameter cannot be used as arg!!!
    # [0.83333333, 1., 0.6, 0.24, 0.072]
    if type(x)==list:
        y=np.zeros(len(x),)
        for i in range(len(x)):
            y[i]=np.math.factorial(x[i])
    else:
        y=np.math.factorial(x)
    return np.exp(-float(lamb))*float(lamb)**np.array(x)/y

def onset_diff(matrix,offset=False,smooth=True,smooth_mode='fuzzy',parameter=parameter): # for dual training option
    assert(set(np.unique(matrix)).issubset({0,1})) #no matter type
    matrix_2=np.zeros(matrix.shape)
    matrix_2[:,0,:]=matrix[:,0,:]
    for i in range(1,matrix.shape[1]):
        matrix_2[:,i,:]=matrix[:,i, :]-matrix[:,i-1, :] #bool works as well
    if not offset:
        matrix_2[matrix_2 == -1] = 0
    else:
        matrix_2=np.abs(matrix_2) # onset+offset
    if smooth:
        matrix_2=onset_smooth(matrix_2,smooth_mode=smooth_mode,parameter=parameter)
    return matrix_2
# def get_onsets(self):
#     """Get all onsets of all notes played by this instrument.
#     May contain duplicates.
#     Returns
#     -------
#     onsets : np.ndarray
#             List of all note onsets.
#     """
#     onsets = []
#     # Get the note-on time of each note played by this instrument
#     for note in self.notes:
#         onsets.append(note.start)
#     # Return them sorted (because why not?)
#     return np.sort(onsets)

def get_onsets_with_notes(notelist,fs,input_len):
    """
    This code is refered to pretty-midi source code
    This is better than calculating the lag of the pianoroll because of the tremolo or legato
    """
    onsets = []
    matrix=np.zeros([128,input_len])
    # Get the note-on time of each note played by this instrument
    for note in notelist:
        onsets.append([note.start,note.pitch])
    # Return them sorted (because why not?)
    onsets=np.array(onsets)
    #onsets=onsets[onsets[:,0].argsort()]

    for i in xrange(onsets.shape[0]):
        if int(onsets[i,0]*fs)<input_len: #
            matrix[int(onsets[i,1]),int(onsets[i,0]*fs)]=1
    return matrix

def post_process_onset(label_onset,label_frame):
    # post-processing for onset-frame result combinaison
    # Reference: CT1 and dual training
    shape_onset=label_onset.shape
    shape_frame=label_frame.shape
    assert(shape_frame==shape_onset)
    label_comb=np.zeros(label_onset.shape)
    ####???????????????????????????????????

    return label_comb
###############################################################################
def framewise_measure_onset(outputs,targets,min_interval = 5): #min_interval depends on sr 
    
    pitch_start_step = {}
    output_note,output_onset = [],[]
    target_note,target_onset = [],[]
    #data form reshape
    for output in outputs:
        output = output.transpose(0,1)
        output_note.append(output[:int(output.shape[0]/2),:,:,:].reshape(output.shape[0]/2,-1))
        output_onset.append(output[int(output.shape[0]/2):,:,:,:].reshape(output.shape[0]/2,-1))
    for target in targets:
        target = target.transpose(0,1)
        target_note.append(target[:int(target.shape[0]/2),:,:,:].reshape(target.shape[0]/2,-1))
        target_onset.append(target[int(target.shape[0]/2):,:,:,:].reshape(target.shape[0]/2,-1))
    output_note = np.concatenate(output_note,1).transpose(1,0)
    output_onset = np.concatenate(output_onset,1).transpose(1,0)
    target_note = np.concatenate(target_note,1).transpose(1,0)   
    target_onset = np.concatenate(target_onset,1).transpose(1,0)   

    output_note = np.append(output_note, [np.zeros(output_note[0].shape)], 0)
    output_onset = np.append(output_onset, [np.zeros(output_onset[0].shape)], 0)
    target_note = np.append(target_note, [np.zeros(target_note[0].shape)], 0)
    # Ensure that any frame with an onset prediction is considered active.
    output_note = np.logical_or(output_note, output_onset)

#    def process_active_pitch(pitch, i):
#        """Process a pitch being active in a given frame."""
#        if pitch not in pitch_start_step:
#            # If onset predictions were supplied, only allow a new note to start
#            # if we've predicted an onset.
#            if output_onset[i, pitch]:
#                pitch_start_step[pitch] = i          
#            else:
#                # Even though the frame is active, the onset predictor doesn't
#                # say there should be an onset, so ignore it.
#                output_note[i,pitch] = 0
#        else:
#            # pitch is already active, but if this is a new onset, we should end
#            # the note and start a new one.
#            if (output_onset[i, pitch] and
#                not output_onset[i - 1, pitch]):
#                end_pitch(pitch, i)
#                pitch_start_step[pitch] = i
##
#    def end_pitch(pitch, end_frame):
#        """End an active pitch."""
#        if (end_frame - pitch_start_step[pitch]) < min_interval:
##            print end_frame - pitch_start_step[pitch]
#            output_note[pitch_start_step[pitch]:end_frame+1,pitch] = 0
#        del pitch_start_step[pitch]
##
#    for i, frame in enumerate(output_note):
#        for pitch, active in enumerate(frame):
#            if active:
#                process_active_pitch(pitch, i)
#            elif pitch in pitch_start_step:
#                end_pitch(pitch, i)
                
    pos_recall=target_note.sum()
    pos_precision=output_note.sum()
    TP=(target_note*output_note).sum()
    
    return [1,TP.item(), pos_precision.item(), pos_recall.item()] 
    


            


    
    
    
    