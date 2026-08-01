# 📖 Quran Recitation Classification - Deep Learning Web App & REST API

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

An end-to-end Deep Learning system and interactive web platform that identifies and classifies Quran reciter voices (Qaris) from raw audio files (`WAV`, `MP3`, `OGG`, `M4A`, `FLAC`) or live microphone recordings using a customized 2D Convolutional Neural Network (CNN) and ensemble audio chunking.

---

## 🌟 Key Features

- **2D-CNN Audio Classifier**: Built with PyTorch to process 2D Log-Mel Spectrogram representations (`128 x 256`).
- **Ensemble Sliding Window Chunking**: Segments long audio files into overlapping 5-second windows, runs inference across all chunks, and averages predicted probabilities for maximum classification accuracy.
- **Dynamic "Skip Intro" Offset**: Features a user-adjustable slider (0–15 seconds) to automatically bypass introductory recitations (such as *Bismillah*).
- **Silence Trimming**: Built-in audio preprocessing using `librosa.effects.trim` with a `top_db=30` threshold.
- **Streamlit Interactive Dashboard (`app.py`)**:
  - Drag-and-drop file uploader (`WAV`, `MP3`, `OGG`, `M4A`, `FLAC`).
  - Live browser microphone recorder.
  - Sleek reciter result card displaying Arabic & English names, bio, and confidence percentage.
  - Real-time Log-Mel Spectrogram visualization via Matplotlib.
  - Interactive probability distribution bar chart powered by Plotly.
- **FastAPI REST Service (`api.py`)**: Production-ready `/predict` HTTP POST endpoint for mobile/web integrations with automatic Swagger UI documentation.
- **Docker Containerization (`Dockerfile`)**: Pre-configured Linux container bundling Python 3.10, PyTorch, and `FFmpeg` for hassle-free deployment.

---

## 🧠 Deep Learning Architecture & How It Works

### 1. Audio Preprocessing & Spectrogram Generation
1. **Audio Loading**: The input file is loaded at a standard sample rate of **22,050 Hz**.
2. **Silence Removal**: Leading and trailing silence are stripped (`top_db=30`).
3. **Intro Offset**: An offset parameter (default 5.0 seconds) trims common introductory phrases (*Bismillah*).
4. **Mel-Spectrogram Transformation**: The audio is converted into a Log-Mel Spectrogram using an `n_fft` of 2,048, a `hop_length` of 512, and 128 Mel frequency bins.
5. **Fixed Resizing**: Spectrograms are resized to a uniform `(128, 256)` grid (`Height=128`, `Width=256`).

### 2. 2D Convolutional Neural Network (`Net`)
The custom PyTorch 2D-CNN extracts spatial and temporal frequency patterns from the spectrogram:

```
Input Spectrogram Tensor: (Batch_Size, 1, 128, 256)
        │
        ▼
[ Conv2D (1 -> 16, 3x3, pad=1) + ReLU + MaxPool2D (2x2) ]   ──► Output: (16, 64, 128)
        │
        ▼
[ Conv2D (16 -> 32, 3x3, pad=1) + ReLU + MaxPool2D (2x2) ]  ──► Output: (32, 32, 64)
        │
        ▼
[ Conv2D (32 -> 64, 3x3, pad=1) + ReLU + MaxPool2D (2x2) ]  ──► Output: (64, 16, 32)
        │
        ▼
[ Flatten ]                                                ──► 64 * 16 * 32 = 32,768 units
        │
        ▼
[ Linear (32,768 -> 1,024) + ReLU + Dropout (0.3) ]
        │
        ▼
[ Linear (1,024 -> 256) + ReLU + Dropout (0.3) ]
        │
        ▼
[ Linear (256 -> 12) ]                                     ──► Logits for 12 Qaris
```

---

## 📊 Training Performance & Benchmark Statistics

The model was trained on Kaggle ([`mohammedalrajeh/quran-recitations-for-audio-classification`](https://www.kaggle.com/datasets/mohammedalrajeh/quran-recitations-for-audio-classification)) over 24 epochs using an Adam optimizer.

| Metric | Score / Value |
| :--- | :--- |
| **Dataset** | 12 Renowned Qaris Audio Corpus |
| **Training Epochs** | 24 Epochs |
| **Training Duration** | ~447 Seconds |
| **Final Training Loss** | `0.1155` |
| **Final Training Accuracy** | **`99.32%`** |
| **Validation Loss** | `0.7447` |
| **Validation Accuracy** | **`97.51%`** |
| **Test Set Prediction Accuracy** | **`97.31%`** |

---

## 🎙️ Supported Reciters (12 Qaris)

| # | Reciter Name (English) | Reciter Name (Arabic) | Title / Information |
|:---:|:---|:---|:---|
| **1** | **Abdul Bari Al-Thubaity** | الشيخ عبد الباري الثبيتي | Imam and Khateeb of Al-Masjid an-Nabawi in Madinah |
| **2** | **Abdul Rahman Al-Sudais** | الشيخ عبد الرحمن السديس | General President for the Two Holy Mosques & Imam of Masjid al-Haram |
| **3** | **Abdullah Al-Buaijan** | الشيخ عبد الله البعيجان | Imam and Khateeb of Al-Masjid an-Nabawi in Madinah |
| **4** | **Ali Al-Huthaify** | الشيخ علي عبد الرحمن الحذيفي | Chief Imam of Al-Masjid an-Nabawi in Madinah |
| **5** | **Bandar Baleela** | الشيخ بندر بليلة | Imam and Khateeb of Masjid al-Haram in Makkah |
| **6** | **Maher Al-Muaiqly** | الشيخ ماهر المعيقلي | Renowned Imam of Masjid al-Haram in Makkah |
| **7** | **Muhammad Al-Luhaidan** | الشيخ محمد اللحيدان | Prominent Qari and Judge from Saudi Arabia |
| **8** | **Muhammad Ayyub** | الشيخ محمد أيوب | Late former Imam of Al-Masjid an-Nabawi |
| **9** | **Nasser Al-Qatami** | الشيخ ناصر القطامي | Famous Qari and Imam in Riyadh |
| **10** | **Saad Al-Ghamdi** | الشيخ سعد الغامدي | World-renowned Qari and former Imam of Masjid an-Nabawi |
| **11** | **Saud Al-Shuraim** | الشيخ سعود الشريم | Former long-time Imam & Khateeb of Masjid al-Haram |
| **12** | **Yasser Al-Dosari** | الشيخ ياسر الدوسري | Imam and Khateeb of Masjid al-Haram in Makkah |

---

## 🚀 Quick Start (Local Setup)

### 1. Clone Repository & Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/faizzanasghar/QuranReciterClassification.git
cd QuranReciterClassification

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Streamlit Web Application

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run FastAPI REST API Service

```bash
python api.py
# or
uvicorn api:app --reload --port 8000
```
Interactive API docs will be available at `http://localhost:8000/docs`.

---

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t quran-reciter-ai .

# Run container
docker run -p 8501:8501 quran-reciter-ai
```

---

## 📁 Repository Structure

```
QuranReciterClassification/
├── app.py                             # Streamlit Interactive Web Application
├── api.py                             # FastAPI REST API Service
├── model.py                           # PyTorch CNN Architecture & Reciter Metadata
├── predict.py                         # Preprocessing, Sliding Window & Inference
├── qurankareemaudioclassification.ipynb # Jupyter Notebook (Kaggle Training Pipeline)
├── requirements.txt                   # Python dependencies list
├── Dockerfile                         # Production Docker container blueprint
├── .gitignore                         # Git ignore configuration
└── README.md                          # Comprehensive Documentation
```

---

## 👤 Author Information

**Muhammad Faizan Asghar**
- **GitHub**: [@faizzanasghar](https://github.com/faizzanasghar)
- **Repository**: [QuranReciterClassification](https://github.com/faizzanasghar/QuranReciterClassification)
- **Domain**: Machine Learning / Deep Learning for Audio Signal Processing
