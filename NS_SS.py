# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 00:18:27 2025

@author: abhsingh
"""

import librosa 
import soundfile as sf
import numpy as np
'''
Attenuation=-20
def spectral_subtraction(audio,sr):
    stft=librosa.stft(audio,n_fft=512,hop_length=256,window='hann')
    mag,phase=np.abs(stft),np.angle(stft)
    noise_est=np.mean(mag[:,:6],axis=1)
    energy=np.sum(mag**2,axis=0)
    threshold=np.max(energy) *(10**(Attenuation/10))
    clean=np.zeros_like(mag)
    silence=energy<threshold
    
    for t in range(mag.shape[1]-1):
        subtracted=mag[:,t]-2*noise_est[:]
        clean[:,t]=np.maximum(subtracted,0)
        if silence[t]:
            clean[:,t]*=(10**(Attenuation/20))
    clean_sig=clean*np.exp(1j*phase)
    clean_audio=librosa.istft(clean_sig,n_fft=512,hop_length=256)
    return clean_audio
    
    
        
    
signal,sr=librosa.load("ip.wav",sr=16000)
clean_signal=spectral_subtraction(signal,16000)
sf.write('clean_2.wav',data=clean_signal,samplerate=sr)
'''
import numpy as np
import soundfile as sf  # to read audio files

def autocorrelation_pitch(y, sr, fmin=50, fmax=500):
    """
    Estimate pitch using autocorrelation.
    y    : audio samples (mono)
    sr   : sample rate
    fmin : minimum expected frequency (Hz)
    fmax : maximum expected frequency (Hz)
    """
    # Normalize signal
    y = y - np.mean(y)
    
    # Autocorrelation
    corr = np.correlate(y, y, mode='full')
    corr = corr[len(corr)//2:]  # keep positive lags
    
    # Limit the search range for lags based on fmin/fmax
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    
    # Find the peak in the allowed range
    corr_peak_index = np.argmax(corr[0:max_lag]) #+ min_lag
    pitch_freq = sr / corr_peak_index
    
    return pitch_freq

# Example usage:
# Load audio (make sure it's mono)
audio, sr = sf.read("ip.wav")
if audio.ndim > 1:
    audio = audio[:, 0]  # take first channel if stereo

pitch = autocorrelation_pitch(audio, sr)
print(f"Estimated pitch: {pitch:.2f} Hz")
