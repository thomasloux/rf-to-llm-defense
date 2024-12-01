import gradio as gr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import torch
from scipy.io.wavfile import write
from dotenv import load_dotenv
import os
from mistralai import Mistral

from denoise import denoise, get_speech, radio_noise

# np.array [length] nb_seconds*freq (44000)
def get_model():
    model_id = "openai/whisper-large-v3"
    if torch.cuda.is_available():
        device = "cuda"
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
    y = radio_noise(y, add_gaussian=True)
    
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
    return transcriber({"sampling_rate": sr, "raw": stream})["text"]


def total_pipeline(messages, audio):
    """
    args:
    list: privous sumaries
    audio: new audio chunk
    """
    transcription = transcribe(audio)
    if len(messages) == 0:
        prompt = f"""
You are a radio operator in the military. Your role is to transcribe and summarize messages from text. You will be provided message, one message per <<>> delimiter. First provide a global summary of messages received (one bullet point per message).
 For the last communication only, you need to provide the following information if provided only, using previous messages as context.
- Date: (Date of the message.)
- Time: (Timestamp of the message.)
- Category of message: (The type of message being sent, for instance "Securite", "Danger", "Mayday", "Ennemi Presence", "Injured Soldier", "Contact with Ennemi".)
- Reason for call: (The purpose of the transmission.)
- Sender: (Information about the sender.)
- Destination: (The recipient of the message.)
- Position: (Coordinates or location details.)
- Task: (What will be done by the sender.)
- Ennemi presence: (Details about the enemy presence and its vehicules.)
- Urgency level: (The urgency of the message, if mentioned.)
"""
        messages = [{"role": "system", "content": prompt}]
    client = Mistral(api_key=os.getenv("API_KEY"))
    model = "mistral-large-latest"

    messages.append({"role": "user", "content": f"<<{transcription}>>"})
    response = client.chat.complete(
        model=model,  # ou un autre modèle disponible
        messages=messages,
    )
    resp = (response.choices[0].message.content)
    # messages.append({"role": "assistant", "content": resp})

    return messages, resp, transcription, "\n".join([m["content"] for m in messages])


def toggle_visibility(all_summaries, isVisible):
    """Toggle the visibility of the 'all_summaries' textbox."""
    return gr.update(visible=not isVisible), not isVisible


if __name__ == "__main__":
    load_dotenv() # os.getenv("API_KEY")

    transcriber = get_model()
    gr.set_static_paths(paths=["assets/"])
    image_path = "assets/logo.jpeg"
    # recording = gr.Audio(sources=["microphone"])

    # demo = gr.Interface(
    #     transcribe,
    #     [recording],
    #     ["text"],
    #     title=logo_html
    # )
    with gr.Blocks() as demo:
        # with gr.Row():  # Add a row for the logo
        #     gr.Image(image_path, label="Logo")
        with gr.Row():  # Add the audio input and text output
            recording = gr.Audio(sources=["microphone"], label="Record Your Audio")
        with gr.Row():
            transcription = gr.Textbox(label="Transcription")
        with gr.Row():
            messages = gr.State(value=[])
            output = gr.Markdown(label="Rapport actuel")
        with gr.Row():
            all_summaries = gr.Textbox(label="Rapport complet", visible=False)
        with gr.Row():
            show_button = gr.Button("Show Full Report")
            is_visible = gr.State(value=False)
            show_button.click(toggle_visibility, inputs=[all_summaries, is_visible], outputs=[all_summaries, is_visible])


        # messages = []
        # Button for triggering the transcription
        btn = gr.Button("Transcribe")
        btn.click(total_pipeline, inputs=[messages, recording], outputs=[messages, output, transcription, all_summaries])

    # Launch the app
    demo.launch()
