
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import numpy as np
from faster_whisper import WhisperModel

st.set_page_config(layout="wide")
st.title("🔊 Transcrição e Espectrograma com Faster Whisper")

uploaded_file = st.file_uploader("Faça upload de um arquivo de áudio", type=["wav", "mp3", "m4a"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(tmp_path)

    st.text("🔁 Carregando modelo Faster Whisper...")
    model = WhisperModel("base", compute_type="int8")

    st.text("🧠 Transcrevendo com timestamps...")
    segments, info = model.transcribe(tmp_path, word_timestamps=True)

    st.text("🎼 Carregando áudio e gerando espectrograma...")
    y, sr = librosa.load(tmp_path)
    S = librosa.stft(y)
    S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    words = []
    for segment in segments:
        for word in segment.words:
            words.append({
                "start": word.start,
                "end": word.end,
                "text": word.word
            })

    # Busca de palavra
    search_word = st.text_input("🔍 Buscar palavra na transcrição (sensível a maiúsculas/minúsculas)")
    highlighted_times = [w["start"] for w in words if search_word in w["text"]] if search_word else []

    # Zoom
    zoom_min = st.slider("⏱️ Tempo inicial (segundos)", 0.0, float(len(y) / sr), 0.0, 0.5)
    zoom_max = st.slider("⏱️ Tempo final (segundos)", zoom_min + 0.5, float(len(y) / sr), float(len(y) / sr), 0.5)

    fig, ax = plt.subplots(figsize=(16, 6))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    ax.set_xlim(zoom_min, zoom_max)
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.title("🎙️ Espectrograma com palavras")

    for word in words:
        if word["start"] >= zoom_min and word["start"] <= zoom_max:
            color = "yellow" if search_word and search_word in word["text"] else "white"
            ax.axvline(x=word["start"], color="red", linestyle="--", alpha=0.6)
            ax.text(word["start"], sr / 4, word["text"], rotation=45, color=color, fontsize=8)

    st.pyplot(fig)
