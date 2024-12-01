import gradio as gr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import torch
from scipy.io.wavfile import write
from dotenv import load_dotenv
import os
from mistralai import Mistral

from denoise import denoise, get_speech, radio_noise


# Paramètres pour extraire les données
target_info = "Sender"
target_value = "Alpha-3 Actual"
target_search = False


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
    You are a radio operator in the military. Your role is to transcribe and summarize messages from text. For each communication in the input file, fill out the report using the following guidelines:
    - Date: (Date of the message.)
    - Time: (Timestamp of the message.)
    - Category of message: (The type of message being sent, for instance "Securite", "Danger", "Mayday", "Ennemi Presence", "Injured Soldier", "Contact with Ennemi".)
    - Reason for call: (The purpose of the transmission, if mentioned.)
    - Sender: (Information about the sender, if mentioned.)
    - Destination: (The recipient: entity or people, if mentioned.)
    - Position: (Coordinates or location details, if mentioned.)
    - Frequency: (Specify radio frequencies, if mentioned.)
    - Order: (Identify all instructions or commands given in the message, if mentioned.)
    - Task: (What will be done by the sender, if mentioned.)
    - Ennemi presence: (Details about the enemy presence and its vehicules, if mentioned.)
    - Urgency level: (The urgency of the message, if mentioned.)

    Leave a blank if the information is not available.
    """
    messages = [{"role": "system", "content": prompt}] #messages = [{"role": "user", "content": prompt}]
    client = Mistral(api_key=os.getenv("API_KEY"))
    model = "mistral-large-latest"

    messages.append({"role": "user", "content": f"<<{transcription}>>"})
    response = client.chat.complete(
        model=model,  # ou un autre modèle disponible
        messages=messages,
    )
    resp = (response.choices[0].message.content)
    # messages.append({"role": "assistant", "content": resp})

    # Extraire le contenu brut de la réponse
    raw_response = chat_response.choices[0].message.content
    clean_response = nettoyer_reponse(raw_response)

    # Diviser les réponses en sections par double nouvelle ligne
    responses = clean_response.strip().split("\n\n")

    # Charger les données existantes
    data = charger_json()
    # Ajouter les nouvelles transmissions au dictionnaire existant
    for i, response in enumerate(responses):
        # Extraire les données de chaque communication
        new_transmission = extraire_donnees(response) 

        # Extraire la date/heure et le sender de la nouvelle transmission
        actual_date = new_transmission.get('Date')  
        actual_time = new_transmission.get('Time')  
        actual_sender = new_transmission.get('Sender')

        # Vérifier si une transmission avec la même date/heure et sender existe déjà
        duplicate_found = False
        for key, existing_transmission in data.items():
            existing_date = existing_transmission.get('Date')  # Récupérer la date de l'existant
            existing_time = existing_transmission.get('Time')  # Récupérer la date de l'existant
            existing_sender = existing_transmission.get('Sender')  # Récupérer le sender de l'existant
            if existing_date == actual_date and existing_time==actual_time and existing_sender == actual_sender:
                duplicate_found = True
                break  # Si une duplication est trouvée, sortir de la boucle

        # Si aucune duplication n'est trouvée, ajouter la transmission
        if not duplicate_found:
            # Ajouter la nouvelle transmission sous une clé unique
            transmission_key = f"transmission_{len(data) + 1}"
            data[transmission_key] = new_transmission
        else:
            print(f"Duplicate transmission found for {actual_date}, {actual_time} and {actual_sender}, skipping.")

    # Enregistrer les données mises à jour dans le fichier JSON
    enregistrer_json(data)
    extraction_reponse(target_search, data)
    
    return messages, resp, transcription, "\n".join([m["content"] for m in messages])


def toggle_visibility(all_summaries, isVisible):
    """Toggle the visibility of the 'all_summaries' textbox."""
    return gr.update(visible=not isVisible), not isVisible



# Charger le texte à partir du fichier
def lire_fichier(fichier):
    try:
        with open(fichier, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{fichier}' n'a pas été trouvé.")
        return None

def charger_json(fichier="outputs_data.json"):
    try:
        with open(fichier, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Erreur : Le fichier JSON '{fichier}' n'a pas été trouvé. Un nouveau fichier sera créé.")
        # Créer un fichier JSON vide si il n'existe pas
        with open(fichier, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=4, ensure_ascii=False)
        return {}  # Retourner un dictionnaire vide

# Enregistrer les résultats dans un fichier JSON
def enregistrer_json(data, fichier="outputs_data.json"):
    try:
        with open(fichier, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Les données ont été enregistrées dans '{fichier}'.")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du fichier JSON : {e}")

# Fonction pour extraire les données du markdown
def extraire_donnees(communication):
    # Initialiser un dictionnaire pour chaque communication
    transmission = {
        "Date": "",
        "Time": "",
        "Category of message": "",
        "Reason for call": "",
        "Sender": "",
        "Destination": "",
        "Position": "",
        "Frequency": "",
        "Order": "",
        "Task": "",
        "Ennemi presence": "",
        "Urgency level": ""
    }

    # Expressions régulières pour extraire les champs
    patterns = {
        "Date": r"Date:\s*(.*?)(?:\n|$)",
        "Time": r"Time:\s*(.*?)(?:\n|$)",
        "Category of message": r"Category of message:\s*(.*?)(?:\n|$)",
        "Reason for call": r"Reason for call:\s*(.*?)(?:\n|$)",
        "Sender": r"Sender:\s*(.*?)(?:\n|$)",
        "Destination": r"Destination:\s*(.*?)(?:\n|$)",
        "Position": r"Position:\s*(.*?)(?:\s*-\s*|$)",
        "Frequency": r"Frequency:\s*(.*?)(?:\s*-\s*|$)",
        "Order": r"Order:\s*(.*?)(?:\n|$)",
        "Task": r"Task:\s*(.*?)(?:\n|$)",
        "Ennemi presence": r"Ennemi presence:\s*(.*?)(?:\n|$)",
        "Urgency level": r"Urgency level:\s*(.*?)(?:\n|$)"
    }

    # Pour chaque clé, extraire la valeur correspondante avec l'expression régulière
    for key, pattern in patterns.items():
        match = re.search(pattern, communication)
        if match:
            transmission[key] = match.group(1).strip()

    return transmission

# Nettoyer la réponse pour supprimer les astérisques Markdown (ou autres caractères indésirables)
def nettoyer_reponse(response):
    # Supprimer les astérisques autour des éléments
    cleaned_response = response.replace("**", "")
    return cleaned_response

def extraction_reponse(target_search, data_set):
    if target_search:
        # Extraire les messages venant de "motif"
        group_motif = {
            key: value
            for key, value in data_set.items()
            if value.get(target_info) == target_value
        }
        # Afficher les résultats
        print(json.dumps(group_motif, indent=4))   
    









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
