import gradio as gr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import torch
from denoise import denoise


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


def transcribe(stream, new_chunk):
    """
    Stream every half minute
    """
    print(new_chunk[1].shape, new_chunk[0])
    sr, y = new_chunk
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
        
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
    return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]


def process(stream, new_chunk):
    sr, y = new_chunk
    

def to_transcribe(stream):
    sr, y = stream
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    return transcriber({"sampling_rate": sr, "raw": y})["text"]


if __name__ == "__main__":
    transcriber = get_model()

    recording = gr.Audio(sources=["microphone"], streaming=True)

    demo = gr.Interface(
        transcribe,
        ["state", recording],
        ["state", "text"],
        live=True,
        stream_every=2
    )

    demo.launch()
