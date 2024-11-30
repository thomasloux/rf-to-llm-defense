import noisereduce as nr
import numpy as np
import scipy



def denoise(signal : np.array, frqcy : int ) -> np.array: 
    nperseg=int(frqcy) # with timesteps of 1s
    _, _,  Sxx = scipy.signal.spectrogram(signal, frqcy, nperseg=nperseg, noverlap=0)
    sum=Sxx.sum(axis=0) # getting the power of signal at this time step
    sum= sum/np.max(sum)
    sum=np.concatenate([sum, np.zeros(1)])
    voice_only_speak=np.array([signal[i] for i in range(len(signal)) if sum[int(i/nperseg)]>0.1 ])
    noise_only=np.array([signal[i] for i in range(len(signal)) if sum[int(i/nperseg)]<0.1 ])
    return nr.reduce_noise(y=voice_only_speak, sr=frqcy, y_noise=noise_only)
    