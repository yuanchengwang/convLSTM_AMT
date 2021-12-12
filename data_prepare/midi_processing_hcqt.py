#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 14:35:30 2018

@author: wyc

"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

Preprocessing for midi-to-npy i.e, 130000_dataset 
logic: midi-> data cleaning(eliminate the broken files+select the midi including the instruments we need) 
->pretty_midi.fliudsyn(soundfonts.sf2)->audio sequence -> CQT+label .npy

WARNING: this code will automaticly remove the broken and not match midi file
         please install fluidsynth before running this code
"""
import os,sys
import librosa as lb
import pretty_midi
import numpy as np
from multiprocessing import Pool
from functools import partial
sys.path.append("./data_prepare")
import HCQT
import time
import onset_utils as ou
#paths selecting
#file_path_mid = '/home/wyc/Desktop/130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]'
file_path_mid = '/home/wyc/Desktop/midi_hcqt'
file_path_save = '/home/wyc/Desktop'
#soundfonts
sf_path = '/usr/share/sounds/sf2/'
sf_name = ['grand-piano-YDP-20160804.sf2','FluidR3_GM.sf2']
#parameters
RangeMIDInotes=[21,108]
CQTMIDInotes=[21,104]
n_worker = 3
sr = 44100
bins_per_octave = 36
n_octave = 7
harmonics = [1,2,3,4] #[0.5,1,2,3,4,5]
dual_train=True
onset_mode='stack'
smooth_mode='fuzzy'
single_chan=True

#Selected instrument
instrument_index={'woodwind':[60,68,70,71,73], #bassoon, clarinet, horn,flute and oboe 
'piano':[0,1,2,3,4,5,6,7], # piano-type
'RWC_similar':[40,64,65,66,67,70,71]} #bassoon, clarinet, violin, sax
#REF website: http://www.ccarh.org/courses/253/handout/gminstruments/

     
def mid2wav (midi_path,sf_name,file_path_save,file_path_mid) : 
#transfer midi file to wav and save
#Need to install fluidsyn, please modify the code if you wanna use Pretty Midi Fluidsyn
    try:                
        b = file_path_save + '/wav' + midi_path[len(file_path_mid):-(len(midi_path.split('/')[-1][:])+1)]
        b_full = file_path_save + '/wav' + midi_path[len(file_path_mid):-3] + 'wav'
        if not os.path.exists(b):
            os.makedirs(b)
        if not os.path.exists(b_full):  # skip the synthesized files
            os.system("fluidsynth /usr/share/sounds/sf2/%s -F %s %s" % (sf_name,b_full,midi_path))
            print b_full
    except:        
        os.system("rm %s" % (midi_path))
        print 'drop:broken' 
    
def mid2seq_npz(midi_path,sr,instrument_index,sf_path,sf_name,file_path_save): #transfer midi to numpy array
    woodwind = instrument_index['woodwind']
    piano = instrument_index['piano']
    RWC_similar = instrument_index['RWC_similar']
    sf_name_piano = sf_name[0]
    sf_name_general = sf_name[1]

    try:
        type_woodwind = midiselection(midi_path,woodwind)
        type_piano = midiselection(midi_path,piano)
        type_RWC_similar = midiselection(midi_path,RWC_similar)
    
        if type_woodwind == 1: 
            b = file_path_save + '/woodwind' 
            b_full = file_path_save + '/woodwind/' + midi_path.split('/')[-1][:-4] + '_seq'                    
            if not os.path.exists(b):
                os.makedirs(b)
            if not os.path.exists(b_full + '.npz'):  # skip the synthesized files
                midi_data = pretty_midi.PrettyMIDI(midi_path)
                audio_data = midi_data.fluidsynth(sr,sf_path + sf_name_general)  
                np.savez_compressed(b_full,audio_data)                    
                print b_full          
            else:
                print 'exists in woodwind'
    
        elif type_piano == 1: 
            b = file_path_save + '/piano' 
            b_full = file_path_save + '/piano/' + midi_path.split('/')[-1][:-4] + '_seq'                     
            if not os.path.exists(b):
                os.makedirs(b)
            if not os.path.exists(b_full + '.npz'):  # skip the synthesized files
                midi_data = pretty_midi.PrettyMIDI(midi_path)
                audio_data = midi_data.fluidsynth(sr,sf_path + sf_name_piano)  
                np.savez_compressed(b_full,audio_data)                    
                print b_full          
            else:
                print 'exists in piano'
    
        elif type_RWC_similar == 1: #and type_piano == 0: 
            b = file_path_save + '/RWC_similar' 
            b_full = file_path_save + '/RWC_similar/' + midi_path.split('/')[-1][:-4] + '_seq'                     
            if not os.path.exists(b):
                os.makedirs(b)
            if not os.path.exists(b_full + '.npz'):  # skip the synthesized files
                midi_data = pretty_midi.PrettyMIDI(midi_path)
                audio_data = midi_data.fluidsynth(sr,sf_path + sf_name_general)                 
                np.savez_compressed(b_full,audio_data)                    
                print b_full              
            else:
                print 'exists in RWC_similar'
        else:            
            print 'drop:not match'
            os.system("rm %s" % (midi_path))
    except:
        print 'drop:broken' 
        os.system("rm %s" % (midi_path))        

def mid2cqt_label(midi_path,sr,bins_per_octave,n_octave,RangeMIDInotes,CQTMIDInotes,instrument_index,sf_path,sf_name,file_path_save,hcqt,dual_train=True,onset_mode='concat',smooth=True,smooth_mode='fuzzy',single_chan=True,**hcqt_args): #transfer midi to numpy array
        n_bins = n_octave*bins_per_octave
        woodwind = instrument_index['woodwind']
        piano = instrument_index['piano']
        RWC_similar = instrument_index['RWC_similar']
        sf_name_piano = sf_name[0]
        sf_name_general = sf_name[1]

#   try:
        type_woodwind = midiselection(midi_path,woodwind)
        type_piano = midiselection(midi_path,piano)
        type_RWC_similar = midiselection(midi_path,RWC_similar)
    
        if type_woodwind == 1: 
            b = file_path_save + '/woodwind' 
            b_full_cqt = file_path_save + '/woodwind/' + midi_path.split('/')[-1][:-4] + '_cqt' 
            b_full_label = file_path_save + '/woodwind/' + midi_path.split('/')[-1][:-4] + '_label'         
            if not os.path.exists(b):
                os.makedirs(b)
            if not os.path.exists(b_full_cqt + '.npz'):  # skip the synthesized files
                midi_data = pretty_midi.PrettyMIDI(midi_path)
                audio_data = midi_data.fluidsynth(sr,sf_path + sf_name_general)                 
                if hcqt == False:
                    CQT_spectrum = lb.cqt(audio_data,sr,fmin = lb.midi_to_hz(min(CQTMIDInotes)),n_bins = n_bins,bins_per_octave = bins_per_octave)
                    CQT = np.expand_dims(np.transpose(np.abs(CQT_spectrum)),axis=0)
                else:
                    CQT_spectrum = HCQT.compute_hcqt(audio_data,sr,FMIN = lb.midi_to_hz(min(CQTMIDInotes)),FMAX = lb.midi_to_hz(max(CQTMIDInotes)),BINS_PER_OCTAVE = bins_per_octave,**hcqt_args)
                    CQT = CQT_spectrum.transpose((0,2,1))
                midi_label=mid2label(midi_path, len(audio_data), CQT.shape[1], sr, RangeMIDInotes=RangeMIDInotes,dual_train=dual_train,onset_mode=onset_mode,smooth=smooth,smooth_mode=smooth_mode)
                if single_chan:
                    midi_label=ou.comb_dim(midi_label) #single channel output
                if dual_train and smooth and smooth_mode!='delay':
                    np.savez_compressed(b_full_label, midi_label)
                else:
                    np.savez_compressed(b_full_label,midi_label.astype('bool'))
                print b_full_label
                np.savez_compressed(b_full_cqt,CQT)                    
                print b_full_cqt
            else:
                print 'exist in woodwind'
    
        elif type_piano == 1: 
            b = file_path_save + '/piano' 
            b_full_cqt = file_path_save + '/piano/' + midi_path.split('/')[-1][:-4] + '_cqt' 
            b_full_label = file_path_save + '/piano/' + midi_path.split('/')[-1][:-4] + '_label'         
            if not os.path.exists(b):
                os.makedirs(b)
            if not os.path.exists(b_full_cqt + '.npz'):  # skip the synthesized files
                midi_data = pretty_midi.PrettyMIDI(midi_path)
                audio_data = midi_data.fluidsynth(sr,sf_path + sf_name_piano)                 
                if hcqt == False:
                    CQT_spectrum = lb.cqt(audio_data,sr,fmin = lb.midi_to_hz(min(CQTMIDInotes)),n_bins = n_bins,bins_per_octave = bins_per_octave)
                    CQT = np.expand_dims(np.transpose(np.abs(CQT_spectrum)),axis=0) 
                else:
                    CQT_spectrum = HCQT.compute_hcqt(audio_data,sr,FMIN = lb.midi_to_hz(min(CQTMIDInotes)),FMAX = lb.midi_to_hz(max(CQTMIDInotes)),BINS_PER_OCTAVE = bins_per_octave,**hcqt_args)
                    CQT = CQT_spectrum.transpose((0,2,1))
                midi_label=mid2label(midi_path, len(audio_data), CQT.shape[1], sr, RangeMIDInotes=RangeMIDInotes,dual_train=dual_train,onset_mode=onset_mode,smooth=smooth,smooth_mode=smooth_mode)
                if single_chan:
                    midi_label=ou.comb_dim(midi_label) #single channel output
                if dual_train and smooth and smooth_mode!='delay':
                    np.savez_compressed(b_full_label, midi_label)
                else:
                    np.savez_compressed(b_full_label,midi_label.astype('bool'))
                print b_full_label
                np.savez_compressed(b_full_cqt,CQT)
                print b_full_cqt
            else:
                print 'exist in piano'
    
        elif type_RWC_similar == 1: 
            b = file_path_save + '/RWC_similar' 
            b_full_cqt = file_path_save + '/RWC_similar/' + midi_path.split('/')[-1][:-4] + '_cqt' 
            b_full_label = file_path_save + '/RWC_similar/' + midi_path.split('/')[-1][:-4] + '_label'         
            if not os.path.exists(b):
                os.makedirs(b)
            if not os.path.exists(b_full_cqt + '.npz'):  # skip the synthesized files
                midi_data = pretty_midi.PrettyMIDI(midi_path)
                audio_data = midi_data.fluidsynth(sr,sf_path + sf_name_general)                 
                if hcqt == False:
                    CQT_spectrum = lb.cqt(audio_data,sr,fmin = lb.midi_to_hz(min(CQTMIDInotes)),n_bins = n_bins,bins_per_octave = bins_per_octave)
                    CQT = np.expand_dims(np.transpose(np.abs(CQT_spectrum)),axis=0)
                else:
                    CQT_spectrum = HCQT.compute_hcqt(audio_data,sr,FMIN = lb.midi_to_hz(min(CQTMIDInotes)),FMAX = lb.midi_to_hz(max(CQTMIDInotes)),BINS_PER_OCTAVE = bins_per_octave,**hcqt_args)
                    CQT = CQT_spectrum.transpose((0,2,1))
                midi_label=mid2label(midi_path, len(audio_data), CQT.shape[1], sr, RangeMIDInotes=RangeMIDInotes,dual_train=dual_train,onset_mode=onset_mode,smooth=smooth,smooth_mode=smooth_mode)
                if single_chan:
                    midi_label=ou.comb_dim(midi_label) #single channel output
                if dual_train and smooth and smooth_mode!='delay':
                    np.savez_compressed(b_full_label, midi_label)
                else:
                    np.savez_compressed(b_full_label,midi_label.astype('bool'))
                print b_full_label
                np.savez_compressed(b_full_cqt,CQT)                    
                print b_full_cqt
            else:
                print 'exist in RWC_similar'
        else:            
            print 'drop:not match'
            os.system("rm %s" % (midi_path))
#    except:
#        print 'drop:broken' 
#        os.system("rm %s" % (midi_path))
        

def mid2label(midi_path, length, CQT_len, sr, RangeMIDInotes=RangeMIDInotes,dual_train=True,onset_mode='concat',smooth=True,smooth_mode='fuzzy'):
    instrument_index = []
    Ground_truth_mats = []
    onset_mats=[]
    midi_data = pretty_midi.PrettyMIDI(midi_path)  
    for instrument in midi_data.instruments:
        instrument_index.append(instrument.program)        
        pianoRoll = instrument.get_piano_roll(fs=CQT_len * sr/length)                      
        if CQT_len < pianoRoll.shape[1]:            
            Ground_truth_mat = (pianoRoll[RangeMIDInotes[0]:RangeMIDInotes[1] + 1, :CQT_len] > 0)
        else:
            Ground_truth_mat = np.concatenate([pianoRoll[RangeMIDInotes[0]:RangeMIDInotes[1] + 1, :],np.zeros([88,(CQT_len-pianoRoll.shape[1])])],1)
        if dual_train:  # for onset label
            onset_matrix = ou.onset_matrix(instrument, CQT_len * sr / length, Ground_truth_mat.shape[1], #automatic zero_padding
                                           smooth=smooth,smooth_mode=smooth_mode)  # change the extra parameters in onset utils!!!
            onset_matrix = onset_matrix[RangeMIDInotes[0]:RangeMIDInotes[1] + 1, :]
            if onset_mode == 'alternative_stack': # chans_instrument×2 alternative channel
                Ground_truth_mat=np.asarray([Ground_truth_mat, onset_matrix]).transpose((0, 2, 1))  #[88,win_width]->[2,win_width,88]
            elif onset_mode == 'concat':
                Ground_truth_mat =np.transpose(np.concatenate([Ground_truth_mat, onset_matrix], axis=0))   # [88,win_width]->[win_width,176]
            else:#'isolated'
                Ground_truth_mat=np.transpose(Ground_truth_mat)
                onset_mat=np.transpose(onset_matrix)  # [win_width,88]
                onset_mats.append(onset_mat)
        Ground_truth_mats.append(Ground_truth_mat)
    if dual_train and onset_mode=='isolated':
        return np.array(Ground_truth_mats),np.array(onset_mats) #[instrument_chans,win_width,88]*2
    elif dual_train and onset_mode == 'alternative_stack':
        return np.concatenate(Ground_truth_mats,axis=0)
    elif dual_train and onset_mode == 'full_stack':
        return np.concatenate([np.array(Ground_truth_mats), np.array(onset_mats)],axis=0)#[instrument_chans*2,win_width,88] [GGGGOOOO]
    else:  # isolated
        return np.array(Ground_truth_mats) #[instrument_chans,win_width,88] or [instrument_chans*2,win_width,88] [GOGOGOGO]


def midiselection(midi_path,instrument_type):   
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    for instrument in midi_data.instruments:   
        instrument_index = instrument.program
        if instrument.is_drum : 
            return 0
        elif not instrument_index in instrument_type :                                 
            return 0           
    return 1

def midiprocessing(file_path_mid,file_path_save,sr,bins_per_octave,n_octave,RangeMIDInotes,CQTMIDInotes, n_worker,instrument_index,sf_path,sf_name,form='cqt_label',
                   hcqt = False,dual_train=True,onset_mode='concat',smooth=True,smooth_mode='fuzzy',single_chan=True,**hcqt_args):
#file_path_mid: the path of mid files
#form: transfer to which form ['wav','cqt_label','seq_npz']
#hcqt: True|hcqt , False|cqt
    if dual_train:
        assert (onset_mode == 'alternative_stack' or onset_mode == 'full_stack' or onset_mode == 'concat' or onset_mode == 'isolated')
        if smooth:
            assert(smooth_mode=='fuzzy' or smooth_mode=='delay' or smooth_mode=='poisson')
    assert(form=='wav' or form=='cqt_label' or form == 'seq_npz')   
    midi_paths = []               
    for root,dirs,files in os.walk(file_path_mid): 
        for file in files:
            midi_paths.append(os.path.join(root,file))
    print 'files number: %s' % (len(midi_paths))
                          
    if n_worker == 1:
        for a in midi_paths:
            if form == 'wav':           
                mid2wav(a,sf_name[1],file_path_save,file_path_mid)
            elif form == 'seq_npz':                
                mid2seq_npz(a,sr,instrument_index,sf_path,sf_name,file_path_save)
            else:
                mid2cqt_label(a,sr,bins_per_octave,n_octave,RangeMIDInotes,CQTMIDInotes,instrument_index,sf_path,sf_name,file_path_save,hcqt,dual_train=dual_train,
                              onset_mode=onset_mode,smooth=smooth,smooth_mode=smooth_mode,single_chan=True,**hcqt_args)
    else:
        if form == 'wav':
            pool = Pool(processes = n_worker) 
            partial_mid2wav=partial(mid2wav,sf_name=sf_name[1],file_path_save=file_path_save,file_path_mid=file_path_mid)
            _ = pool.map(partial_mid2wav,midi_paths)
            pool.close()
            pool.join()
        elif form == 'seq_npz':
            pool = Pool(processes = n_worker)
            partial_mid2seq_npz = partial(mid2seq_npz,sr = sr,instrument_index = instrument_index,sf_path = sf_path,sf_name = sf_name,file_path_save = file_path_save)
            _ = pool.map(partial_mid2seq_npz,midi_paths)
            pool.close()
            pool.join()            
        else:
            pool = Pool(processes = n_worker)
            partial_mid2cqt_label = partial(mid2cqt_label,sr = sr,bins_per_octave = bins_per_octave ,n_octave = n_octave,RangeMIDInotes = RangeMIDInotes,CQTMIDInotes = CQTMIDInotes,
                                            instrument_index = instrument_index,sf_path = sf_path,sf_name = sf_name,file_path_save = file_path_save,hcqt = hcqt,dual_train=dual_train,onset_mode=onset_mode,smooth=smooth,smooth_mode=smooth_mode,single_chan=True,**hcqt_args)
            _ = pool.map(partial_mid2cqt_label,midi_paths)
            pool.close()
            pool.join()
      
if __name__=='__main__':
    start = time.time()
    #if hcqt = True
    midiprocessing(file_path_mid,file_path_save,sr=sr, bins_per_octave=bins_per_octave, n_octave=n_octave,
                    RangeMIDInotes=RangeMIDInotes,CQTMIDInotes=CQTMIDInotes,n_worker = n_worker,instrument_index=instrument_index,sf_path=sf_path,sf_name=sf_name,
                    form = 'cqt_label',hcqt = True,dual_train=True,onset_mode='concat',smooth=True,smooth_mode=smooth_mode,single_chan=True,
                    harmonics = harmonics,mode = 'reshape',output_form = 'norm')
    #if hcqt = False
#    midiprocessing(file_path_mid,file_path_save,sr=sr, bins_per_octave=bins_per_octave, n_octave=n_octave,RangeMIDInotes=RangeMIDInotes,
#                   CQTMIDInotes=CQTMIDInotes,n_worker = n_worker,instrument_index=instrument_index,sf_path=sf_path,sf_name=sf_name,
#                   form = 'cqt_label',hcqt = False,dual_train=True,onset_mode='concat',smooth=True,smooth_mode=smooth_mode,single_chan=True)
    end = time.time()
    print 'finished in %ss'% (end-start)



