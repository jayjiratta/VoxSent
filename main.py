import streamlit as st
import tensorflow as tf
import whisper
import tempfile
import os
import numpy as np
import pickle
from keras.models import load_model
import json

# import json file for emotions
with open("emotions.json", "r", encoding="utf-8") as file:
    emotions_data = json.load(file)

# Load whisper model
whisper_model = whisper.load_model("base")

# Build sidebar for navigation and selected models
st.sidebar.title("Navigation")
if st.sidebar.button("Home"):
    app_mode = "Home"
elif st.sidebar.button("About Project"):
    app_mode = "About Project"
else:
    app_mode = "Home" 

st.sidebar.title("Settings")
MODEL_OPTIONS = {
    "28 Emotion Analysis": "version1",
    "7 Emotion Analysis": "version2"
}
selected_model_name = st.sidebar.selectbox(
    "Select Emotion Model",
    list(MODEL_OPTIONS.keys()) 
)

emotion_model_version = MODEL_OPTIONS[selected_model_name]

@st.cache_resource
# Load emotion model
def load_emotion_model(version):
    model_path = f"models/{version}/emotion_prediction_model.h5"
    tokenizer_path = f"models/{version}/tokenizer.pickle"

    model = load_model(model_path)

    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    return model, tokenizer

model, tokenizer = load_emotion_model(emotion_model_version)

# Function to predict emotion
def predict_emotion(text, model, tokenizer, label_map):
    sequences = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
    predictions = model.predict(padded)[0]

    predicted_class = np.argmax(predictions)
    return label_map.get(predicted_class, "Unknown")


# Label maps for different emotion models
LABEL_MAPS = {
    "version1": {25: 'sadness', 0: 'admiration', 27: 'neutral', 18: 'love', 15: 'gratitude', 
                10: 'disapproval', 1: 'amusement', 9: 'disappointment', 22: 'realization', 
                3: 'annoyance', 6: 'confusion', 20: 'optimism', 7: 'curiosity', 
                13: 'excitement', 5: 'caring', 11: 'disgust', 24: 'remorse', 17: 'joy', 
                4: 'approval', 12: 'embarrassment', 26: 'surprise', 2: 'anger', 
                16: 'grief', 21: 'pride', 8: 'desire', 23: 'relief', 14: 'fear', 
                19: 'nervousness'},

    "version2": {0: 'anger', 1: 'disgust', 2: 'neutral', 3: 'surprise', 
                4: 'joy', 5: 'sadness', 6: 'fear'}
}

label_map = LABEL_MAPS[emotion_model_version]

# Function to transcribe audio
def transcribe_audio(audio_file):
    try:
        result = whisper_model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        return ""

if(app_mode=="Home"):
    st.title("Speech & Text Emotion Analysis")

    st.header("Upload or Record Audio")
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
    temp_audio_path = None
    transcribed_text = ""

    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            temp_audio_path = temp_audio.name

        try:
            st.audio(temp_audio_path, format=audio_file.type)
            transcribed_text = transcribe_audio(temp_audio_path)
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
        finally:
            os.remove(temp_audio_path)

    st.subheader("Record Audio")
    audio_value = st.audio_input("Record a voice message")

    if audio_value:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_record:
            temp_record.write(audio_value.read())
            temp_record_path = temp_record.name

        try:
            st.audio(temp_record_path, format=audio_value.type)
            transcribed_text = transcribe_audio(temp_record_path)
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
        finally:
            os.remove(temp_record_path)


    st.text_area("Transcribed Text", transcribed_text, height=100, disabled=True)

    text_input = st.text_area("Enter text for analysis", transcribed_text, height=100)
    analyze_button = st.button("Analyze")

    if analyze_button and text_input:
        predicted_emotion = predict_emotion(text_input, model, tokenizer, label_map)

        # Fetch emoji and text from JSON
        emotion_data = emotions_data.get(predicted_emotion, {"emoji": "😐", "text": "Emotion not found"})
        emoji = emotion_data["emoji"]
        emotion_text = emotion_data["text"]

        # Display emotion with emoji and text
        st.subheader("Emotion Analysis")
        st.markdown(
            f"""
            <div style="text-align: center;">
                <span style="font-size: 60px;">{emoji}</span>
                <h2 style="margin-top: 5px;">{predicted_emotion.upper()}</h2>
                <p style="border-top: 2px solid #ddd; padding-top: 10px; font-size: 18px;">
                    {emotion_text}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


elif(app_mode=="About Project"):
    st.markdown("""
### About Project

This project is a **Speech & Text Emotion Analysis** system built using **Streamlit**, **TensorFlow**, and **Whisper**. It processes both audio and text inputs, transcribes speech into text, and analyzes emotions using deep learning models.

#### Dataset and Model Training
- **Speech-to-Text (STT) Model:** Uses OpenAI's **Whisper** for speech transcription.
- **Emotion Models:** Supports two versions:
  - **28-Emotion Model** for detailed analysis.
  - **7-Emotion Model** for generalized classification.
- **Training Process:**
  - Preprocessed text data using tokenization and padding.
  - LSTM-based deep learning models trained on large emotion-labeled datasets.
  - Optimized using cross-entropy loss and Adam optimizer.

#### Features
- 🎤 **Speech-to-Text Processing** using Whisper.
- 📊 **Emotion Classification** from text or speech.
- 🔄 **Multiple Models:** Switch between 28-emotion and 7-emotion models.
- 🎙️ **Real-Time Audio Input** for instant analysis.
- 🚀 **User-Friendly Interface** built with Streamlit.

This project aims to integrate **multimodal emotion recognition**, making it useful for applications in research, AI interactions, and customer service.
""")
