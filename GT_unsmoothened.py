# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 16:21:07 2025

@author: abhsingh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:44:17 2025

@author: abhsingh
"""

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from scipy.signal import butter, lfilter
from scipy.signal import welch
import wave
import numpy as np
import scipy.io.wavfile as wav
import os


def apply_fadin(audio):
    length=len(audio)
    audio=list(audio/32767.0)
    p=int(length/5)
    fade_curve = np.linspace(.4, 1.0,p )
    mm = audio[:p] * fade_curve
    audio[:p] =mm
    audio1=np.array(audio)
    audio1=audio1*32767
    audio1=audio1.astype("int16")
    return audio1



def apply_fadeout(audio):
    length = len(audio)
    audio=list(audio/32767.0)
    p=int(length/5)
   # start = end - length
    fade_curve = np.linspace(1.0, .4, p)
    mm = audio[-p:] * fade_curve
    audio[-p:] =mm
    audio1=np.array(audio)
    audio1=audio1*32767
    audio1=audio1.astype("int16")
    return audio1



def wav_read(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(n_frames)

        dtype = np.int16 if sample_width == 2 else np.int32
        audio = np.frombuffer(audio_data, dtype=dtype)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
            audio = audio[:, 0]
        return audio, frame_rate
    
    
def wav_write(file_name, data, sample_rate):
    with wave.open(file_name, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data.astype(np.int16).tobytes())
    
def _psd(signal, sample_rate):
    freqs, psd = welch(signal, fs=sample_rate, nperseg=repeat_samples[str(frame_duration)],noverlap=0)
    return psd,freqs
 
input_folder = "left"
frame_duration =0.0160#0.032 #0.16 #0.080

os.makedirs("outputs", exist_ok=True)

   
#input_file="Woman_PP_Clean_5sec.wav"
initial_frame={"0.16":1,"0.08":2,"0.032":3,"0.016":2}
repeat_samples={"0.16":2560,"0.08":1280,"0.032":512,"0.016":256}

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".wav"):
        continue

    input_path = os.path.join(input_folder, filename)
    name_prefix = os.path.splitext(filename)[0]
    
    clean_signal, sample_rate_clean = wav_read(input_path)
    
    frame_size = int(frame_duration * sample_rate_clean)
    length=len(clean_signal)
    num_frames = (length ) // (frame_size)
    
    if(length % frame_size):
        remainder=length % frame_size
        zero_pad=repeat_samples[str(frame_duration)]-remainder
        num_frames+=1
        length=length+zero_pad
    
    clean_signal = np.pad(clean_signal, (0, zero_pad), 'constant')
    clean_signal1=np.array(clean_signal)/np.mean(clean_signal)
    
    k1_mean=[]
    VAD=[]
    VAD_repeat=[]
    speech_segments=[]
    non_speech_segments=[]
    for i in range(0,num_frames):
        start = i * frame_size
        end = start + frame_size
        if end > length:
            break
        clean=clean_signal1[start:end]
        psd_frame_clean_signal, freqs_frame = _psd(clean, sample_rate_clean)
        psd_frame_clean_signal=psd_frame_clean_signal#[0:160]
        psd_mean_clean=np.mean(psd_frame_clean_signal)
        k1_mean.append(psd_mean_clean)
    

    'considering initial frames as stationary noise/silence'
    
    noise_floor = np.mean(k1_mean[:initial_frame[str(frame_duration)]]) # 3 for 512,2 for 1280,1 for 2560 FS
    
    
    psd_threshold = noise_floor * 200 # set 200 for 1280 and 200 for 2560 FS
    
    for i in range(0,num_frames):
        start = i * frame_size
        end = start + frame_size
        if end > length:
            break
        clean = clean_signal1[start:end]
        clean_ref = clean_signal[start:end]
        psd_mean_clean = k1_mean[i]  # PSD mean
        if psd_mean_clean > psd_threshold:
            VAD.append(1)
           # out = apply_fadin(clean_ref)# not required in clean speech
           # out = apply_fadeout(out) #not required in clean speech
            speech_segments.append(clean_ref)
        else:
            VAD.append(0)
            #out = apply_fadin(clean_ref)
            #out = apply_fadeout(out)
            non_speech_segments.append(clean_ref)
    


    with open(f"outputs/{name_prefix}_VADDecision.txt", 'w') as f:
        for i, vad_value in enumerate(VAD, 1):
            f.write(f"frame{i}: {vad_value}  Time:{(i*256-256)/16000} sec\n")
            
            
    k1=np.repeat(k1_mean,repeat_samples[str(frame_duration)])
    VAD_repeat=np.repeat(VAD,repeat_samples[str(frame_duration)])
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6)) 
    
    ax1.plot(clean_signal, label="WavFile")
    ax1.legend()
    
    ax2.plot(VAD_repeat, label="VAD")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"outputs/{name_prefix}_.png", dpi=300) 
    plt.show()
    
    
    wav_write(f"outputs/{name_prefix}_VADDecision.wav", np.array(VAD_repeat), sample_rate_clean)
    wav_write(f"outputs/{name_prefix}_PSD.wav", np.array(k1), sample_rate_clean)
    wav_write(f"outputs/{name_prefix}_Speech.wav",  np.array(speech_segments), sample_rate_clean)
    wav_write(f"outputs/{name_prefix}_NonSpeech.wav",  np.array(non_speech_segments), sample_rate_clean)

