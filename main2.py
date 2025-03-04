import streamlit as st
from transformers import pipeline
import whisper
import tempfile
import os
import numpy as np
import soundfile as sf
from datetime import datetime, timedelta
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings

if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Navigation functions
def navigate_to(page):
    st.session_state.page = page
    st.rerun()

# Load models
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

emotion_pipeline = load_emotion_model()
whisper_model = load_whisper_model()

emotion_emojis = {"joy": "😁", "anger": "😡", "sadness": "😢", "fear": "😨", "surprise": "😲", "love": "❤️", "neutral": "😐"}

def transcribe_audio(audio_file):
    try:
        result = whisper_model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        return ""

# Sidebar navigation menu
with st.sidebar:
    st.title("Navigation")
    
    # Always visible menu items
    if st.button("Home"):
        navigate_to('home')
    
    if st.button("About Me"):
        navigate_to('about')

# Main content area based on current page
if st.session_state.page == 'home':
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
            if temp_audio_path and os.path.exists(temp_audio_path):
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
            if 'temp_record_path' in locals() and os.path.exists(temp_record_path):
                os.remove(temp_record_path)
        
    st.text_area("Transcribed Text", transcribed_text, height=100, disabled=True)

    text_input = st.text_area("Enter text for analysis", transcribed_text, height=100)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        analyze_button = st.button("Analyze")
    
    if analyze_button and text_input:
        emotion_result = emotion_pipeline(text_input)
        top_emotion = max(emotion_result, key=lambda x: x['score'])
        emotion = top_emotion["label"].lower()
        
        st.subheader("Emotion Analysis")
        st.markdown(f"**Emotion:** {emotion_emojis.get(emotion, '😐')} ({emotion})")

elif st.session_state.page == 'about':
    st.title("About Me")
    st.write("This is the About Me page. Here you can add information about yourself or the project.")
    st.write("This application uses Whisper for speech recognition and a RoBERTa model for emotion analysis.")
    
    st.subheader("Technologies Used")
    st.markdown("""
    - Streamlit for the web interface
    - Hugging Face Transformers for emotion analysis
    - OpenAI Whisper for speech recognition
    """)
    
    st.subheader("Contact Information")
    st.write("You can add your contact information here.")
