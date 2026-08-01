import os
import tempfile
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from predict import predict_reciter
from model import RECITERS_INFO

# Page Configuration
st.set_page_config(
    page_title="Quran Reciter Identification AI",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    /* Theme Colors & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .arabic-text {
        font-family: 'Amiri', serif;
        direction: rtl;
        text-align: right;
    }

    /* Main Container Card */
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #f4f5f6;
    }

    .main-header p {
        font-size: 1.1rem;
        color: #d1d5db;
    }

    /* Reciter Result Card */
    .reciter-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }

    .reciter-name-ar {
        font-family: 'Amiri', serif;
        font-size: 2.3rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0.3rem;
    }

    .reciter-name-en {
        font-size: 1.6rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.8rem;
    }

    .confidence-badge {
        display: inline-block;
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }

    .reciter-desc {
        color: #94a3b8;
        font-size: 1rem;
        line-height: 1.5;
    }

    /* Sidebar Reciter List */
    .sidebar-reciter-item {
        padding: 0.6rem 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    .stAudio {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def create_probability_chart(probabilities):
    """Creates a sleek horizontal bar chart of reciter probabilities using Plotly."""
    names_en = [item['name_en'] for item in probabilities.values()]
    names_ar = [item['name_ar'] for item in probabilities.values()]
    probs = [item['probability'] * 100 for item in probabilities.values()]

    # Reverse order for top-to-bottom bar display
    names_display = [f"{en} ({ar})" for en, ar in zip(names_en, names_ar)][::-1]
    probs = probs[::-1]

    colors = ['#38bdf8' if i == len(probs)-1 else '#475569' for i in range(len(probs))]

    fig = go.Figure(go.Bar(
        x=probs,
        y=names_display,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
        ),
        text=[f"{p:.1f}%" for p in probs],
        textposition='outside'
    ))

    fig.update_layout(
        title=dict(
            text="Reciter Probability Breakdown",
            font=dict(size=18, color="#f8fafc")
        ),
        xaxis=dict(
            title="Confidence Probability (%)",
            range=[0, 115],
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(color='#94a3b8')
        ),
        yaxis=dict(
            tickfont=dict(size=13, color='#f8fafc')
        ),
        margin=dict(l=20, r=20, t=50, b=30),
        height=480,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def plot_spectrogram(spectrogram):
    """Plots log-mel spectrogram using matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='none')
    ax.set_facecolor('none')
    im = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title("Resized Log-Mel Spectrogram (128 x 256)", color='#f8fafc', fontsize=12, pad=10)
    ax.set_xlabel("Time Frames", color='#94a3b8')
    ax.set_ylabel("Mel Frequency Bins", color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='#94a3b8')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94a3b8')
    fig.tight_layout()
    return fig


def main():
    # Sidebar
    st.sidebar.title("📖 Reciters Catalog")
    st.sidebar.markdown("This AI model classifies recitations from **12 renowned Qaris**:")

    for idx, (key, info) in enumerate(RECITERS_INFO.items(), 1):
        st.sidebar.markdown(f"**{idx}. {info['en']}**  \n<span class='arabic-text' style='color:#38bdf8;'>{info['ar']}</span>", unsafe_allow_html=True)
        st.sidebar.markdown("---")

    st.sidebar.title("⚙️ Settings")
    offset_sec = st.sidebar.slider(
        "Skip Intro (seconds)",
        min_value=0.0,
        max_value=15.0,
        value=5.0,
        step=0.5,
        help="Skip the beginning of the audio (e.g., 'Bismillah') to focus on the core recitation."
    )

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Quran Reciter Classification AI</h1>
        <p>Upload or record an audio clip of Quran recitation to automatically identify the Qari using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)

    # Checkpoint status warning
    if not os.path.exists("quran_audio_model.pt"):
        st.warning("""
        ⚠️ **Trained Weights (`quran_audio_model.pt`) Not Found!**  
        The app is currently using a **randomly initialized (untrained) PyTorch model**, which causes inaccurate predictions.
        - **If you trained the model on Kaggle or Google Colab**: Download your `quran_audio_model.pt` file and place it directly into this project directory (`e:\\Documents\\ML Projects\\QuranRecitationClassification\\`).
        """, icon="⚠️")

    # Input Mode Tabs
    tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎙️ Record Audio"])

    audio_file_buffer = None

    with tab1:
        uploaded_file = st.file_uploader(
            "Choose an audio clip (WAV, MP3, OGG, M4A, FLAC)",
            type=["wav", "mp3", "ogg", "m4a", "flac"]
        )
        if uploaded_file is not None:
            audio_file_buffer = uploaded_file

    with tab2:
        if hasattr(st, "audio_input"):
            recorded_audio = st.audio_input("Record audio clip of recitation")
            if recorded_audio is not None:
                audio_file_buffer = recorded_audio
        else:
            st.info("Audio recording input is supported in newer Streamlit versions. Please upload an audio file.")

    # Process Prediction
    if audio_file_buffer is not None:
        st.markdown("### 🔊 Audio Playback")
        st.audio(audio_file_buffer)

        with st.spinner("Analyzing audio features & running neural network inference..."):
            try:
                result = predict_reciter(audio_file_buffer, offset_sec=offset_sec)
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                result = None

        if result is not None:
            st.markdown("---")
            st.markdown("## 🎯 Classification Result")
    
            col1, col2 = st.columns([1, 1])
    
            with col1:
                # Result Card
                st.markdown(f"""
                <div class="reciter-card">
                    <div class="reciter-name-ar">{result['predicted_name_ar']}</div>
                    <div class="reciter-name-en">{result['predicted_name_en']}</div>
                    <div class="confidence-badge">Confidence: {result['confidence']*100:.1f}%</div>
                    <div class="reciter-desc">{result['description']}</div>
                    <div style="margin-top: 10px; font-size: 0.9rem; color: #64748b;">
                        Analyzed {result.get('num_chunks', 1)} overlapping audio chunks.
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
                # Spectrogram Plot
                st.markdown("### 📊 Audio Mel-Spectrogram")
                spec_fig = plot_spectrogram(result['spectrogram'])
                st.pyplot(spec_fig)
    
            with col2:
                # Probability Bar Chart
                st.markdown("### 📈 Reciter Probabilities")
                prob_fig = create_probability_chart(result['probabilities'])
                st.plotly_chart(prob_fig, use_container_width=True)

    else:
        st.info("👋 Please upload an audio file or record a recitation clip above to begin classification.")


if __name__ == "__main__":
    main()
