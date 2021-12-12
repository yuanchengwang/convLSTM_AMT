# this configuration only works for conv mode, valid conv by default, 45 min,
# net doesn't support dual_training
#config={'model':'conv','dual_train':False,'epoch':30,'batch_size':16,'kernel_size':7,'win_width':32,'n_note':88,'pooling_size':(1,3),'padding_mode':'valid',
#        'input_chans':1,'filter_size1':(5,25),'filter_size2':(3,5),'filter_size3':(1,24),'num_features1':50,'num_features2':50,'num_features3':1000,'num_features4':500}

#valid with dual train activated
#config={'model':'conv','dual_train':True,'epoch':30,'batch_size':16,'kernel_size':7,'win_width':32,'n_note':176,'pooling_size':(1,3),'padding_mode':'valid',
#        'input_chans':4,'filter_size1':(5,7),'filter_size2':(3,5),'filter_size3':(1,26),'num_features1':80,'num_features2':80,'num_features3':1000,'num_features4':500}


# this configuration only works for conv mode, same conv by default
#config={'model':'conv','dual_train':True,'epoch':30,'batch_size':16,'kernel_size':7,'win_width':32,'n_note':88,'pooling_size':(1,3),'padding_mode':'same',
#        'filter_size1':(5,25),'filter_size2':(3,5),'filter_size3':(1,28),'num_features1':50,'num_features2':50,'num_features3':1000,'num_features4':500}

# this configuration only works for conv mode, valid conv by default,test mod
#support dual train   
#config={'model':'conv','input_chans':4,'dual_train':True,'epoch':30,'batch_size':16,'kernel_size':7,'win_width':32,'n_note':176,'pooling_size':(1,3),'padding_mode':'valid',
#        'filter_size1':(5,7),'filter_size2':(3,5),'filter_size3':(1,26),'num_features1':80,'num_features2':80,'num_features3':1000,'num_features4':500}


# this configuration only works for convlstm valid-type padding for convlstm with conv model, w/o peephole, 35 hours,
# net doesn't support dual_training
#config={'model':'convlstm1','dual_train':False,'epoch':30,'batch_size':16,'kernel_size':7,'n_freq':252,'win_width':32,'n_note':88,'padding_mode':'valid',
#         'pooling_size':(1,1,3),'peephole':False,'Layer_norm':True,'filter_size1':(5,25),'filter_size2':(3,5),'filter_size3':(1,24),'shape2':(3,76),
#         'num_features1':50,'num_features2':50,'num_features3':1000,'num_features4':500}

# this configuration only works for convlstm same-type padding for convlstm with conv model,w peephole, 40 hours,
# net doesn't support dual_training
# config={'model':'convlstm1','dual_train':False,'epoch':12,'batch_size':16,'kernel_size':7,'n_freq':252,'win_width':16,'n_note':88,'padding_mode':'same',
#         'pooling_size':(1,1,3),'peephole':True,'Layer_norm':True,'filter_size1':(5,25),'filter_size2':(3,5),'filter_size3':(1,196),'shape2':(7,84),
#         'num_features1':50,'num_features2':50,'num_features3':1000,'num_features4':500}


# this configuration only works for convlstm same-type padding for convlstm with conv model,w peephole, 40 hours,
# test mode, use it with main_test.py code,input_chans depends on the input form, HCQT supports
#config={'model':'convlstm1','input_chans':5,'dual_train':False,'epoch':12,'batch_size':1,'kernel_size':7,'n_freq':252,'win_width':16,'n_note':88,'padding_mode':'same',
#         'pooling_size':(1,1,3),'peephole':True,'Layer_norm':True,'filter_size1':(5,25),'filter_size2':(3,5),'filter_size3':(1,196),'shape2':(7,84),
#         'num_features1':50,'num_features2':50,'num_features3':1000,'num_features4':500}

# this configuration only works for convlstm same-type padding for convlstm with conv model,w peephole, ** hours,
# support dual_training, use it with main.py code,input_chans depends on the input form, HCQT supports
#config={'model':'convlstm1','input_chans':4,'dual_train':True,'epoch':12,'batch_size':1,'kernel_size':7,'n_freq':252,'win_width':16,'n_note':176,'padding_mode':'same',
#         'pooling_size':(1,1,3),'peephole':True,'Layer_norm':True,'filter_size1':(5,25),'filter_size2':(3,5),'filter_size3':(1,196),'shape2':(7,84),
#         'num_features1':50,'num_features2':50,'num_features3':1000,'num_features4':500}

# this configuration only works for conv and lstm model,w peephole, 6 hours,
# net doesn't support dual_training, use it with main.py code,input_chans depends on the input form, HCQT supports
#config={'model':'conv_lstm','input_chans':4,'dual_train':False,'epoch':30,'batch_size':16,'kernel_size':7,'n_freq':252,'win_width':16,'n_note':88,'padding_mode':'valid',
#         'pooling_size':(1,1,3),'peephole':True,'Layer_norm':True,'filter_size1':(5,25),'filter_size2':(3,5),'filter_size3':(1,24),
#         'num_features1':50,'num_features2':50,'num_features3':1000,'num_features4':500}

# this configuration only works for convlstm and lstm model,w peephole, 38 hours,
# net doesn't support dual_training, use it with main.py code,input_chans depends on the input form, HCQT supports
#config={'model':'convlstm3','input_chans':4,'dual_train':False,'epoch':12,'batch_size':16,'kernel_size':7,'n_freq':252,'win_width':16,'n_note':88,'padding_mode':'same',
#         'pooling_size':(1,1,3),'peephole':True,'Layer_norm':True,'filter_size1':(5,25),'filter_size2':(3,5),'filter_size3':(1,28),'shape2':(7,84),
#         'num_features1':50,'num_features2':50,'num_features3':1000,'num_features4':500}

config={'model':'convlstm3','input_chans':1,'dual_train':'Onset','epoch':12,'batch_size':16,'kernel_size':7,'n_freq':252,'win_width':16,'n_note':88,'padding_mode':'same',
         'pooling_size':(1,1,3),'peephole':True,'Layer_norm':True,'filter_size1':(5,7),'filter_size2':(3,5),'filter_size3':(1,28),'shape2':(7,84),
         'num_features1':50,'num_features2':50,'num_features3':1000,'num_features4':500}


# this configuration only works for convlstm same-type padding for convlstm with CT1 model, with deeper layer,smaller filter, 5 days,
# dual_training can be activated, HCQT is activated since input_chans not =1
#config={'model':'convlstm2','dual_train':True,'epoch':30,'batch_size':16,'kernel_size':7,'n_freq':252,'win_width':16,'n_note':88,'padding_mode':'same',
#        'pooling_size':(1,1,2),'peephole':True,'Layer_norm':True,'filter_size1':(1,3,3),'filter_size2':(1,1,3),'filter_size3':(1,28),'shape2':(7,84),'input_chans':5,
#        'num_features1':50,'num_features2':50,'num_features3':1000,'num_features4':500,'block_num':(2,4,2,1)} # block_num=(Conv,ResConvlstm,NiN,L)

# Configuration for data augmentation
#Aug_config={'amp_aug':True,'amp_range':[-5,5],'pitch_shift':True,'pitch_shift_range':[-1,0,1],'key_aug':True,'key_range':[-6,6],'bins_per_octave':36}
Aug_config={'amp_aug':False,'amp_range':[-5,5],'pitch_shift':False,'pitch_shift_range':[-1,1],'key_aug':False,'key_range':[-5,5],'bins_per_octave':36}