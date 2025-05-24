
import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
from faster_whisper import WhisperModel
import tempfile

st.set_page_config(layout="wide")
st.title("🔊 Transcrição em Linha do Tempo (Waveform + Blocos de Palavra)")

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

    st.text("📈 Carregando waveform e preparando visualização...")
    y, sr = librosa.load(tmp_path, sr=None)
    duration = len(y) / sr
    times = np.linspace(0, duration, num=len(y))

    # Preparar gráfico
    fig = go.Figure()

    # Waveform
    fig.add_trace(go.Scatter(
        x=times,
        y=y,
        mode="lines",
        name="Waveform",
        line=dict(color="royalblue", width=1),
        opacity=0.6,
        hoverinfo="skip"
    ))

    # Palavras
    for segment in segments:
        for word in segment.words:
            fig.add_shape(
                type="rect",
                x0=word.start,
                x1=word.end,
                y0=min(y),
                y1=max(y),
                line=dict(width=0),
                fillcolor="rgba(255,165,0,0.4)",
                layer="below"
            )
            fig.add_trace(go.Scatter(
                x=[(word.start + word.end) / 2],
                y=[max(y) * 0.8],
                text=[word.word],
                mode="text",
                showlegend=False
            ))

    fig.update_layout(
        title="🕒 Linha do Tempo de Transcrição",
        xaxis_title="Tempo (s)",
        yaxis_title="Amplitude",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)
