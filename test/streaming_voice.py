import gradio as gr
import numpy as np
import scipy

### Attempt to have an online version of the voice detection
### The idea would be to have the radio listening continuously and detecting when there is a voice

def noise_signal(signal, noise, add_gaussian=False):
    noise = np.concatenate([noise] * int(len(signal) / len(noise) + 1))
    gaussian = np.random.normal(0, 1, len(signal))
    gaussian = gaussian / np.max(gaussian)
    return signal + 2 * noise[:len(signal)] + (gaussian if add_gaussian else 0) * 0.2


def radio_noise(signal, add_gaussian=False):
    noise_file = "radio_noise.wav"
    noise = scipy.io.wavfile.read("../sounds/" + noise_file)
    noise = noise[1].T[0]
    # Increase volume
    noise = noise / np.max(np.abs(noise))
    noise = noise * 50
    return noise_signal(signal, noise, add_gaussian)


def get_speech(signal: np.array, frqcy: int) -> np.array:

    # Append a speaking part to the signal
    speaking = "../audio/stream.wav"
    speaking = scipy.io.wavfile.read(speaking)
    speaking = speaking[1]
    speaking = speaking / np.max(np.abs(speaking))
    # append the speaking part to the signal
    length_signal = len(signal)
    signal = np.concatenate([signal, speaking])

    # save the signal
    scipy.io.wavfile.write("speech_voice_detection.wav", frqcy, signal)

    nperseg = int(frqcy)  # with timesteps of 1s
    _, _, Sxx = scipy.signal.spectrogram(signal, frqcy, nperseg=nperseg, noverlap=0)
    sum = Sxx.sum(axis=0)  # getting the power of signal at this time step
    sum = sum / np.max(sum)
    sum = np.concatenate([sum, np.zeros(1)])
    # voice_only_speak = np.array(
    #     [signal[i] for i in range(len(signal)) if sum[int(i / nperseg)] > 0.1]
    # )
    # Proportion with more than 10% power
    voice_only_speak = np.mean(sum[:int(length_signal/nperseg)] > 0.2)
    print(voice_only_speak)
    return voice_only_speak > 0.8


def update_transcription(liste_before, audio):
    frq, stream = audio
    print(stream.shape[0]/frq)
    if stream.ndim > 1:
        stream = stream.mean(axis=1)
    stream = stream / np.max(np.abs(stream))

    # Noise it
    stream = radio_noise(stream, add_gaussian=False)

    print(f'Length of stream: {len(liste_before)}')
    if len(liste_before) >= 8:
        liste_before.pop(0)
    liste_before.append(stream)

    stream = np.concatenate(liste_before)

    check_speech = get_speech(stream, frq)
    if check_speech:
        transcription = "Speech detected"
    else:
        transcription = "No speech detected"
    return liste_before, transcription


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():  # Add the audio input and text output
            recording = gr.Audio(
                sources=["microphone"],
                label="Record Your Audio",
                streaming=True,
            )
            # liste_before is list of previous audio
            liste_before = gr.State([])
        with gr.Row():
            transcription = gr.Textbox(label="Transcription")

        recording.stream(
            update_transcription,
            inputs=[liste_before, recording],
            outputs=[liste_before, transcription],
            stream_every=1
            )

    demo.launch()
    
