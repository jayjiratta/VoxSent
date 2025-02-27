import streamlit as st
from transformers import pipeline
import whisper
import tempfile
import os
import numpy as np
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings

# Load models globally
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
whisper_model = whisper.load_model("base")

# Emoji mapping
sentiment_emojis = {"POSITIVE": "😊", "NEGATIVE": "😞", "NEUTRAL": "😐"}
emotion_emojis = {"joy": "😁", "anger": "😡", "sadness": "😢", "fear": "😨", "surprise": "😲", "love": "❤️", "neutral": "😐"}

# Function to process audio
def transcribe_audio(audio_file):
    try:
        result = whisper_model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        return ""

st.title("Speech & Text Sentiment and Emotion Analysis")

# Audio Upload & Recording
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
webrtc_ctx = webrtc_streamer(
    key="speech-record",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    ),
)

if webrtc_ctx.audio_receiver:
    try:
        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=5)
        if audio_frames:
            audio = np.concatenate([frame.to_ndarray() for frame in audio_frames])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio_path = temp_audio.name
                sf.write(temp_audio_path, audio, samplerate=16000)
                
                st.audio(temp_audio_path, format="audio/wav")
                transcribed_text = transcribe_audio(temp_audio_path)
                os.remove(temp_audio_path)
        else:
            st.warning("No audio frames captured. Try speaking longer.")
    except Exception as e:
        st.error(f"Error in recording: {str(e)}")
else:
    st.info("Awaiting microphone access...")

# Display Transcribed Text (Read-Only)
st.text_area("Transcribed Text", transcribed_text, height=100, disabled=True)

# Manual Text Input
text_input = st.text_area("Enter text for analysis", transcribed_text, height=100)
analyze_button = st.button("Analyze")

if analyze_button and text_input:
    sentiment_result = sentiment_pipeline(text_input)[0]
    sentiment = sentiment_result["label"].upper()
    
    emotion_result = emotion_pipeline(text_input)
    top_emotion = max(emotion_result, key=lambda x: x['score'])
    emotion = top_emotion["label"].lower()
    
    st.subheader("Sentiment Analysis")
    st.markdown(f"**Sentiment:** {sentiment_emojis.get(sentiment, '😐')} ({sentiment})")
    
    st.subheader("Emotion Analysis")
    st.markdown(f"**Emotion:** {emotion_emojis.get(emotion, '😐')} ({emotion})")