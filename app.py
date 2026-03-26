import streamlit as st
import librosa
import numpy as np
import pickle
import soundfile
import io

# 1. Load the trained brain you just created
model = pickle.load(open("emotion_model.pkl", "rb"))

# 2. Function to process any new audio file
def extract_feature(file):
    # Load audio and force it to Mono
    X, sample_rate = librosa.load(file, sr=None, mono=True)
    stft = np.abs(librosa.stft(X))
    
    # Get the math features (MFCC, Chroma, Mel)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    
    return np.hstack((mfccs, chroma, mel)).reshape(1, -1)

# 3. Design the Web Page
st.set_page_config(page_title="AI Voice Emotion Detector")
st.title("🎙️ Speech Emotion Recognition")
st.markdown("---")
st.write("Upload a recording from your dataset or your own voice (.wav) to see the AI's prediction!")

# File uploader
uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file is not None:
    # Play the sound so we can hear it
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Analyze Emotion"):
        with st.spinner('The AI is listening...'):
            # Convert audio to features and predict
            features = extract_feature(io.BytesIO(uploaded_file.read()))
            prediction = model.predict(features)
            
            # Show result

            st.success(f"### Predicted Emotion: **{prediction[0].upper()}**")
