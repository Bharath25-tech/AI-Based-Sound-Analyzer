import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import soundfile as sf
import io
import warnings
import time

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Sound Analyzer | AI in Physics",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS – Dark Glassmorphism Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #0a0e1a;
    color: #e8eaf6;
  }

  /* Main background */
  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1529 50%, #0a0e1a 100%);
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
  }

  /* Cards */
  .glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
  }

  /* Metric card */
  .metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.08));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
  }
  .metric-card:hover {
    border-color: rgba(99,102,241,0.7);
    box-shadow: 0 0 20px rgba(99,102,241,0.25);
  }
  .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .metric-label {
    font-size: 0.78rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
  }

  /* Hero header */
  .hero {
    text-align: center;
    padding: 2rem 0 1rem;
  }
  .hero h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
  }
  .hero p {
    color: #94a3b8;
    font-size: 1rem;
  }

  /* AI prediction bars */
  .pred-bar-bg {
    background: rgba(255,255,255,0.07);
    border-radius: 8px;
    height: 10px;
    margin-top: 6px;
    overflow: hidden;
  }
  .pred-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    transition: width 0.8s ease;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
    padding: 8px 20px;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #818cf8, #c084fc) !important;
    color: white !important;
  }

  /* File uploader */
  [data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.08);
    border: 2px dashed rgba(99,102,241,0.4);
    border-radius: 12px;
    padding: 1rem;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }

  /* Divider */
  hr { border-color: rgba(255,255,255,0.08) !important; }

  /* Info/warning boxes */
  .stAlert { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f1529",
    "axes.facecolor":    "#0f1529",
    "axes.edgecolor":    "#334155",
    "axes.labelcolor":   "#94a3b8",
    "text.color":        "#e2e8f0",
    "xtick.color":       "#94a3b8",
    "ytick.color":       "#94a3b8",
    "grid.color":        "#1e293b",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "sans-serif",
})

ACCENT = "#818cf8"
ACCENT2 = "#c084fc"

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_audio(file_bytes, sr=22050):
    """Load audio from uploaded bytes."""
    buf = io.BytesIO(file_bytes)
    y, sr = librosa.load(buf, sr=sr, mono=True)
    return y, sr


@st.cache_resource(show_spinner=False)
def load_ai_model():
    """Load HuggingFace audio classification pipeline (cached across reruns)."""
    from transformers import pipeline
    return pipeline(
        "audio-classification",
        model="MIT/ast-finetuned-audioset-10-10-0.4593",
    )


def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    times = np.linspace(0, len(y) / sr, len(y))
    ax.fill_between(times, y, alpha=0.4, color=ACCENT)
    ax.plot(times, y, color=ACCENT, linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Sound Waveform  ·  Amplitude vs. Time", color="#e2e8f0", pad=10)
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_spectrogram(y, sr):
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 3))
    cmap = mcolors.LinearSegmentedColormap.from_list("ag", ["#0f1529", "#818cf8", "#c084fc", "#f0abfc"])
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="log", ax=ax, cmap=cmap)
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="dB")
    ax.set_title("Spectrogram  ·  Short-Time Fourier Transform (STFT)", color="#e2e8f0", pad=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    return fig


def plot_mfcc(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    fig, ax = plt.subplots(figsize=(10, 3))
    cmap = mcolors.LinearSegmentedColormap.from_list("ag2", ["#0f1529", "#0ea5e9", "#818cf8"])
    img = librosa.display.specshow(mfccs, sr=sr, x_axis="time", ax=ax, cmap=cmap)
    fig.colorbar(img, ax=ax, label="MFCC Coefficient")
    ax.set_title("MFCCs  ·  Mel-Frequency Cepstral Coefficients", color="#e2e8f0", pad=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MFCC #")
    fig.tight_layout()
    return fig


def plot_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = range(len(centroid))
    t = librosa.frames_to_time(frames, sr=sr)
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(t, centroid, color=ACCENT2, linewidth=1.5, label="Spectral Centroid")
    ax.fill_between(t, centroid, alpha=0.2, color=ACCENT2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hz")
    ax.set_title("Spectral Centroid  ·  Brightness of Sound Over Time", color="#e2e8f0", pad=10)
    ax.legend(facecolor="#0f1529", edgecolor="#334155")
    ax.grid(True)
    fig.tight_layout()
    return fig


def compute_metrics(y, sr):
    duration      = librosa.get_duration(y=y, sr=sr)
    rms           = float(np.sqrt(np.mean(y ** 2)))
    zcr           = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    f0_arr, _vf, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    f0            = float(np.nanmedian(f0_arr)) if f0_arr is not None else 0.0
    centroid_mean = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth_mean = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    return {
        "Duration (s)":         round(duration, 2),
        "Sample Rate (Hz)":     sr,
        "RMS Energy":           round(rms, 5),
        "Zero Crossing Rate":   round(zcr, 5),
        "Fundamental Freq (Hz)": round(f0, 2),
        "Spectral Centroid (Hz)": round(centroid_mean, 2),
        "Spectral Bandwidth (Hz)": round(bandwidth_mean, 2),
    }


def run_ai_classification(file_bytes, pipe):
    """Run HuggingFace audio classification on raw byte data."""
    buf = io.BytesIO(file_bytes)
    # Load audio at 16kHz (required by AST model)
    y, sr = librosa.load(buf, sr=16000, mono=True)
    # Pass as dict — the correct input format for the HuggingFace pipeline
    audio_input = {"array": y, "sampling_rate": sr}
    results = pipe(audio_input, top_k=5)
    return results


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.2rem 0 0.5rem;'>
      <div style='font-size:3rem;'>🔊</div>
      <div style='font-size:1.3rem; font-weight:700; background: linear-gradient(90deg, #818cf8, #c084fc);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        AI Sound Analyzer
      </div>
      <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>AI in Physics · 2026</div>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    st.markdown("#### 📁 Upload Audio")
    uploaded_file = st.file_uploader(
        "Supported: WAV, MP3, FLAC, OGG",
        type=["wav", "mp3", "flac", "ogg"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.82rem; color:#64748b;'>
    <b style='color:#94a3b8;'>🧠 AI Model</b><br/>
    MIT AST · AudioSet 527-class<br/>
    <br/>
    <b style='color:#94a3b8;'>🔬 Physics Analyses</b><br/>
    • Waveform (Amplitude/Time)<br/>
    • STFT Spectrogram<br/>
    • MFCC Coefficients<br/>
    • Spectral Centroid<br/>
    • Pitch / F₀ Estimation
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <h1>🎵 AI-Based Sound Analyzer</h1>
  <p>Explore the Physics of Sound &amp; Artificial Intelligence — together.</p>
</div>
""", unsafe_allow_html=True)

if uploaded_file is None:
    # Landing state
    col1, col2, col3 = st.columns(3)
    cards = [
        ("🌊", "Waveform Analysis", "Visualize amplitude vs. time — the raw physical motion of sound pressure."),
        ("📊", "Spectral Physics", "Decompose sound into frequencies using the Short-Time Fourier Transform."),
        ("🤖", "AI Classification", "A transformer-based model identifies what type of sound it is from 527 categories."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='text-align:center; min-height:170px;'>
              <div style='font-size:2.5rem;'>{icon}</div>
              <div style='font-size:1rem; font-weight:600; color:#e2e8f0; margin:8px 0 4px;'>{title}</div>
              <div style='font-size:0.82rem; color:#64748b;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; margin-top:2rem; padding:2rem;
                background:rgba(99,102,241,0.07); border-radius:16px;
                border:1px dashed rgba(99,102,241,0.3);'>
      <div style='font-size:2rem;'>⬆️</div>
      <div style='font-size:1.1rem; color:#94a3b8; margin-top:8px;'>
        Upload an audio file from the sidebar to get started
      </div>
      <div style='font-size:0.8rem; color:#475569; margin-top:4px;'>WAV · MP3 · FLAC · OGG supported</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# LOAD AUDIO
# ─────────────────────────────────────────────
file_bytes = uploaded_file.read()
with st.spinner("🔄 Loading audio..."):
    y, sr = load_audio(file_bytes)

# ─────────────────────────────────────────────
# AUDIO PLAYER
# ─────────────────────────────────────────────
st.audio(file_bytes, format=f"audio/{uploaded_file.name.split('.')[-1]}")

# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────
with st.spinner("⚙️ Computing acoustic metrics..."):
    metrics = compute_metrics(y, sr)

metric_keys = list(metrics.keys())
cols = st.columns(len(metric_keys))
for col, key in zip(cols, metric_keys):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value'>{metrics[key]}</div>
          <div class='metric-label'>{key}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_physics, tab_ai = st.tabs(["🔬 Physics Analysis", "🤖 AI Classification"])

# ~~~~ PHYSICS TAB ~~~~
with tab_physics:
    st.markdown("### 🌊 Waveform")
    st.markdown("""
    <div class='glass-card' style='font-size:0.83rem; color:#94a3b8; padding:0.7rem 1rem; margin-bottom:0.5rem;'>
    The waveform shows how the amplitude (displacement of air molecules) varies over time.
    In physics this represents the mechanical wave propagating through the air.
    </div>
    """, unsafe_allow_html=True)
    with st.spinner("Plotting waveform..."):
        st.pyplot(plot_waveform(y, sr), use_container_width=True)

    st.markdown("### 📊 Spectrogram (STFT)")
    st.markdown("""
    <div class='glass-card' style='font-size:0.83rem; color:#94a3b8; padding:0.7rem 1rem; margin-bottom:0.5rem;'>
    The spectrogram applies the <b style='color:#818cf8'>Short-Time Fourier Transform</b> to decompose the sound into its
    constituent frequencies over time. Brighter colors = more energy at that frequency.
    </div>
    """, unsafe_allow_html=True)
    with st.spinner("Computing STFT spectrogram..."):
        st.pyplot(plot_spectrogram(y, sr), use_container_width=True)

    st.markdown("### 🎛️ MFCCs — Mel-Frequency Cepstral Coefficients")
    st.markdown("""
    <div class='glass-card' style='font-size:0.83rem; color:#94a3b8; padding:0.7rem 1rem; margin-bottom:0.5rem;'>
    MFCCs represent sound on a <b style='color:#818cf8'>Mel scale</b> (perceptual frequency scale), closely mimicking
    how the human ear perceives pitch. These are the features fed into AI/ML models.
    </div>
    """, unsafe_allow_html=True)
    with st.spinner("Computing MFCCs..."):
        st.pyplot(plot_mfcc(y, sr), use_container_width=True)

    st.markdown("### ✨ Spectral Centroid (Brightness)")
    st.markdown("""
    <div class='glass-card' style='font-size:0.83rem; color:#94a3b8; padding:0.7rem 1rem; margin-bottom:0.5rem;'>
    The spectral centroid is the <b style='color:#c084fc'>"centre of mass"</b> of the spectrum at each moment in time —
    a higher centroid means the sound is brighter or sharper (e.g., cymbal vs. bass).
    </div>
    """, unsafe_allow_html=True)
    with st.spinner("Computing spectral centroid..."):
        st.pyplot(plot_spectral_centroid(y, sr), use_container_width=True)

# ~~~~ AI TAB ~~~~
with tab_ai:
    st.markdown("### 🤖 AI Audio Classification")
    st.markdown("""
    <div class='glass-card' style='font-size:0.83rem; color:#94a3b8; padding:0.9rem 1.2rem; margin-bottom:1rem;'>
    Using the <b style='color:#818cf8'>Audio Spectrogram Transformer (AST)</b> fine-tuned on Google AudioSet,
    the model classifies the uploaded audio into one of <b style='color:#c084fc'>527 sound categories</b> —
    from speech to music to environmental sounds. This is AI applied to acoustic physics.
    </div>
    """, unsafe_allow_html=True)

    run_btn = st.button("▶  Run AI Classification", use_container_width=True, type="primary")

    if run_btn:
        with st.spinner("🧠 Loading AI model and running inference (first run may take a minute)..."):
            pipe = load_ai_model()
            results = run_ai_classification(file_bytes, pipe)

        st.markdown("#### 🏆 Top Predictions")
        for i, r in enumerate(results):
            label  = r["label"].replace("_", " ").title()
            score  = r["score"]
            width  = int(score * 100)
            medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
            medal  = medals[i] if i < len(medals) else f"#{i+1}"

            st.markdown(f"""
            <div class='glass-card' style='padding:1rem 1.4rem;'>
              <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                  <span style='font-size:1.4rem;'>{medal}</span>
                  <span style='font-size:1rem; font-weight:600; color:#e2e8f0; margin-left:8px;'>{label}</span>
                </div>
                <div style='font-size:1rem; font-weight:700; color:#818cf8;'>{score*100:.1f}%</div>
              </div>
              <div class='pred-bar-bg'>
                <div class='pred-bar-fill' style='width:{width}%;'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.success("✅ Classification complete! The model has analyzed the acoustic features of your audio.")

        st.markdown("#### 📖 How does it work?")
        st.markdown("""
        <div class='glass-card' style='font-size:0.85rem; color:#94a3b8; line-height:1.8;'>
        1. 🎵 <b style='color:#e2e8f0'>Raw audio</b> is loaded at 16 kHz and converted to a <b style='color:#818cf8'>Mel Spectrogram</b> (physics → image).<br/>
        2. 🖼️ The spectrogram image is fed into an <b style='color:#e2e8f0'>Audio Spectrogram Transformer (AST)</b> — a Vision Transformer adapted for audio.<br/>
        3. 🧠 The model was <b style='color:#c084fc'>pre-trained on AudioSet</b> (2M+ labelled clips from YouTube).<br/>
        4. 📊 The output is a probability distribution over <b style='color:#818cf8'>527 AudioSet classes</b> — the highest score wins.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:0.78rem; color:#334155; padding-bottom:1rem;'>
  AI-Based Sound Analyzer &nbsp;·&nbsp; AI in Physics Project &nbsp;·&nbsp; Built with Streamlit, Librosa & HuggingFace Transformers
</div>
""", unsafe_allow_html=True)
