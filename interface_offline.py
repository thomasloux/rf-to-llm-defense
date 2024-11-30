import gradio as gr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import torch
from scipy.io.wavfile import write

from denoise import denoise, get_speech, radio_noise


def get_model():
    model_id = "openai/whisper-large-v3"
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    torch_dtype = torch.float32

    # torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe


def transcribe(new_chunk):
    """
    Stream every half minute
    """
    # print(new_chunk[1].shape, new_chunk[0])
    sr, y = new_chunk

    # Add noise
    # y = radio_noise(y, add_gaussian=True)
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
        
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    stream = y
    # if stream is not None:
    #     stream = np.concatenate([stream, y])
    # else:
    #     stream = y

    speech = get_speech(stream, sr)
    denoised = denoise(stream, sr)
    # save 
    write("stream.wav", sr, stream)
    write("speech.wav", sr, speech)
    write("denoised.wav", sr, denoised)
    return transcriber({"sampling_rate": sr, "raw": denoised})["text"]


if __name__ == "__main__":
    transcriber = get_model()

    recording = gr.Audio(sources=["microphone"])

    demo = gr.Interface(
        transcribe,
        [recording],
        ["text"]
    )

    demo.launch()
