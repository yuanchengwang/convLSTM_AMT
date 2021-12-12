# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 20:52:29 2018
 
@author: 57297
hcqt model based on librosa-cqt
HCQT2.1
"""
import numpy as np
import librosa as lb
from math import log
def compute_hcqt(audio_data,SR,FMIN,FMAX,BINS_PER_OCTAVE,harmonics = [0.5,1,2,3,4,5],mode = 'reshape',output_form = 'norm',zeropadding = True):
    #harmonics: number of harmonics in [0.5,1,2,3,4,5]
    #reshape: calculate hcqt by reshaping a single cqt transform
    #harmonic: calculate hcqt by changing the FMIN
    #norm: normlized output 
    #log: output in log 
    assert(mode == 'reshape' or mode == 'harmonic') 
    assert(output_form == 'norm' or output_form == 'log')
    assert(set(harmonics).issubset({0.5, 1, 2, 3, 4, 5})) #
    assert(min(harmonics) == 0.5 or min(harmonics) == 1)
    #max assert　if　not zero_padding:
    cqt_list = []
    res_h = []
    N_OCTAVES_DIS = int(np.ceil(log(FMAX/FMIN,2)))                                  #number of ocataves:display     
    N_OCTAVES_MAX = int(np.floor(log(SR/2/(FMIN*min(harmonics)),2)))                #number of ocataves:maxium        
    N_OCTAVES_R = int(np.ceil(log(FMAX*max(harmonics)/(FMIN*min(harmonics)),2)))    #number of ocataves:reshape    　
    N_BINS_DIS = int(BINS_PER_OCTAVE/12*(lb.hz_to_midi(FMAX)-lb.hz_to_midi(FMIN)+1))   
    if zeropadding == False:
        assert(N_OCTAVES_R <= N_OCTAVES_MAX)
    if N_OCTAVES_R > N_OCTAVES_MAX:        
        res_r = N_OCTAVES_R-N_OCTAVES_MAX
        N_OCTAVES_R = N_OCTAVES_MAX
    else:
        res_r = 0
        
    for h in harmonics:
        N_OCTAVES_H = int(np.ceil(log(FMAX*h/(FMIN*min(harmonics)),2)))             #number of ocataves:harmonic  
        if N_OCTAVES_H > N_OCTAVES_MAX:        
            res_h.append(N_OCTAVES_H-N_OCTAVES_MAX)           
        else:
            res_h.append(0)        

    if mode == 'reshape':
        cqt = lb.cqt(
            audio_data, sr=SR,fmin=min(harmonics)*FMIN,n_bins = BINS_PER_OCTAVE*N_OCTAVES_R,
            bins_per_octave=BINS_PER_OCTAVE
            )       
        if res_r != 0:
            cqt = np.concatenate([np.abs(cqt),np.zeros([(res_r+1)*BINS_PER_OCTAVE,cqt.shape[1]])],0)
        else:
            cqt = np.abs(cqt)
        if min(harmonics) == 1:
            shift = 1
        else:
            shift = 0
        for h in harmonics:                                
            if h == 0.5 :
                cqt_0_5 = cqt[:N_OCTAVES_DIS*BINS_PER_OCTAVE]
                cqt_list.append(cqt_0_5[:N_BINS_DIS])
            elif h == 1:
                cqt_1 = cqt[(1-shift)*BINS_PER_OCTAVE:(N_OCTAVES_DIS+1-shift)*BINS_PER_OCTAVE]
                cqt_list.append(cqt_1[:N_BINS_DIS])
            elif h == 2:
                cqt_2 = cqt[(2-shift)*BINS_PER_OCTAVE:(N_OCTAVES_DIS+2-shift)*BINS_PER_OCTAVE]
                cqt_list.append(cqt_2[:N_BINS_DIS])
            elif h == 3:                
                cqt_3 = cqt[(2-shift)*BINS_PER_OCTAVE + int(BINS_PER_OCTAVE*7/12) : (N_OCTAVES_DIS+2-shift)*BINS_PER_OCTAVE + int(BINS_PER_OCTAVE*7/12)]
                cqt_list.append(cqt_3[:N_BINS_DIS])
            elif h == 4:
                cqt_4 = cqt[(3-shift)*BINS_PER_OCTAVE:(N_OCTAVES_DIS+3-shift)*BINS_PER_OCTAVE]
                cqt_list.append(cqt_4[:N_BINS_DIS])
            else:                
                cqt_5 = cqt[(3-shift)*BINS_PER_OCTAVE + int(BINS_PER_OCTAVE/3):(N_OCTAVES_DIS+3-shift)*BINS_PER_OCTAVE + int(BINS_PER_OCTAVE/3)]
                cqt_list.append(cqt_5[:N_BINS_DIS])            
    else:
        shapes = []
        for i,h in enumerate(harmonics):
            cqt = lb.cqt(
                audio_data, sr=SR, fmin=FMIN*float(h),
                n_bins=BINS_PER_OCTAVE*(N_OCTAVES_DIS-res_h[i]),
                bins_per_octave=BINS_PER_OCTAVE
                )
            if res_h[i] != 0:
                cqt = np.concatenate([np.abs(cqt),np.zeros([res_h[i]*BINS_PER_OCTAVE,cqt.shape[1]])],0)
            else:
                cqt = np.abs(cqt)
            cqt_list.append(cqt[:N_BINS_DIS])
            shapes.append(cqt.shape)
        
        shapes_equal = [s == shapes[0] for s in shapes]
        if not all(shapes_equal):
            min_time = np.min([s[1] for s in shapes])
            new_cqt_list = []
            for i in range(len(cqt_list)):
                new_cqt_list.append(cqt_list[i][:, :min_time])
            cqt_list = new_cqt_list   
    if output_form == 'norm':        
        hcqt = np.abs(np.array(cqt_list))        
    else:        
        hcqt = ((1.0/80.0) * lb.core.amplitude_to_db(
                np.abs(np.array(cqt_list)), ref=np.max)) + 1.0
    return hcqt 

#test
#SR = 44100
#FMIN = lb.note_to_hz('A0')
#FMAX = lb.note_to_hz('C8')
#BINS_PER_OCTAVE = 36
#harmonics = [0.5,1,2,3,4,5]
#audio_fpath = 'C:/Users/57297/Desktop/bach_846.wav'
#audio_data, fs = lb.load(audio_fpath, sr=SR)
#hcqt = compute_hcqt(audio_data,SR,FMIN,FMAX,BINS_PER_OCTAVE,harmonics,mode = 'reshape',output_form = 'log',zeropadding = True)
#print(hcqt.shape)
#for i in range(hcqt.shape[0]):
#    print(hcqt[i][57][4729])
