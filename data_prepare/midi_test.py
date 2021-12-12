import pretty_midi as pm
import fluidsynth
import librosa as lb
#fluidsynth: warning: Failed to pin the sample data to RAM; swapping is possible. does not matter
#Failed to load SoundFont is a big problem
#error: There is no preset with bank number 0 and preset number 40 in SoundFont 1,
##  bacause of there is not corresponding soundfont


midi_path='/home/wyc/Desktop/09carnivalcuckoo.mid'
sf_path='/usr/share/sounds/sf2/FluidR3_GM.sf2'
#sf_path='/home/wyc/Desktop/grand-piano-YDP-20160804.sf2'
output_path='/home/wyc/Desktop/123.wav'
midi_data = pm.PrettyMIDI(midi_path)
audio_data = midi_data.fluidsynth(44100,sf_path)
lb.output.write_wav(output_path,audio_data,44100)


