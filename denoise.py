import noisereduce as nr
import numpy as np
import scipy
from scipy.io import wavfile



def denoise(signal : np.array, frqcy : int ) -> np.array: 
    nperseg=int(frqcy) # with timesteps of 1s
    _, _,  Sxx = scipy.signal.spectrogram(signal, frqcy, nperseg=nperseg, noverlap=0)
    sum=Sxx.sum(axis=0) # getting the power of signal at this time step
    sum= sum/np.max(sum)
    sum=np.concatenate([sum, np.zeros(1)])
    voice_only_speak=np.array([signal[i] for i in range(len(signal)) if sum[int(i/nperseg)]>0.1 ])
    noise_only=np.array([signal[i] for i in range(len(signal)) if sum[int(i/nperseg)]<0.1 ])
    return nr.reduce_noise(y=voice_only_speak, sr=frqcy, y_noise=noise_only)


def noise_signal(signal, noise, add_gaussian=False): 
    noise=np.concatenate([noise]*int(len(signal)/len(noise)+1))
    gaussian = np.random.normal(0, 1, len(signal))
    gaussian = gaussian/np.max(gaussian)
    return signal + 2*noise[0:len(signal)] + (gaussian if add_gaussian else 0)*0.2



noise_file='radio_noise.wav'
noise=scipy.io.wavfile.read('sounds/'+noise_file)
frqcy_noise=noise[0]
noise=noise[1].T[0]
noise=noise/np.max(np.abs(noise))

    