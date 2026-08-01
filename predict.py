import os
import torch
import torch.nn.functional as F
import numpy as np
import librosa
from skimage.transform import resize
from model import Net, CLASS_NAMES, RECITERS_INFO

DEFAULT_MODEL_PATH = "quran_audio_model.pt"


def extract_mel_spectrogram(audio_source, sr=22050, duration=5, offset_sec=5.0, chunk_overlap=2.5, img_height=128, img_width=256):
    """
    Extracts log-mel spectrogram from an audio file, using a sliding window.
    Supports both file paths and in-memory byte streams.
    """
    import io
    import tempfile
    import os
    
    signal = None
    sample_rate = None
    
    # If Streamlit UploadedFile or BytesIO or raw bytes
    if hasattr(audio_source, "read"):
        audio_source.seek(0)
        audio_bytes = audio_source.read()
        audio_source = io.BytesIO(audio_bytes)
        
        # Determine suffix for temp file fallback
        suffix = getattr(audio_source, "name", ".tmp")
        if isinstance(suffix, str) and "." in suffix:
            suffix = "." + suffix.split(".")[-1]
        else:
            suffix = ".tmp"
    elif isinstance(audio_source, (bytes, bytearray)):
        audio_bytes = audio_source
        audio_source = io.BytesIO(audio_bytes)
        suffix = ".tmp"
    else:
        # String path
        audio_bytes = None
        suffix = None

    try:
        # Try direct load (Works for WAV, OGG, FLAC via soundfile)
        signal, sample_rate = librosa.load(audio_source, sr=sr)
    except Exception as e:
        # Fails for MP3/M4A via soundfile. Audioread requires a real file path.
        if audio_bytes is not None:
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                temp_audio.write(audio_bytes)
                temp_audio.close() # Free Windows file lock
                
                signal, sample_rate = librosa.load(temp_audio.name, sr=sr)
            except Exception as e2:
                raise ValueError(f"Could not decode audio format even with fallback: {e2}")
            finally:
                if os.path.exists(temp_audio.name):
                    os.remove(temp_audio.name)
        else:
            # It was already a path
            raise ValueError(f"Could not decode audio file '{audio_source}': {e}")
            
    # Trim silence
    signal, _ = librosa.effects.trim(signal, top_db=30)
    
    # Skip intro offset
    start_sample = int(offset_sec * sample_rate)
    if start_sample < len(signal):
        signal = signal[start_sample:]
        
    chunk_length = int(duration * sample_rate)
    hop_length = int((duration - chunk_overlap) * sample_rate)
    
    if len(signal) < chunk_length:
        signal = np.pad(signal, (0, chunk_length - len(signal)), mode='constant')
        
    chunks = []
    for start in range(0, len(signal) - chunk_length + 1, hop_length):
        chunks.append(signal[start:start + chunk_length])
        
    if not chunks:
        chunks.append(signal[:chunk_length])
        
    tensors = []
    first_spec = None
    for chunk in chunks:
        spec = librosa.feature.melspectrogram(y=chunk, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        spec_fixed = librosa.util.fix_length(spec_db, size=(duration * sample_rate) // 512 + 1)
        spec_resized = resize(spec_fixed, (img_height, img_width), anti_aliasing=True)
        if first_spec is None:
            first_spec = spec_resized
        tensors.append(torch.tensor(spec_resized, dtype=torch.float32).unsqueeze(0))
        
    # batch_tensor shape: (NumChunks, 1, 128, 256)
    batch_tensor = torch.stack(tensors)
    return batch_tensor, first_spec, signal, sample_rate


def load_model(checkpoint_path=DEFAULT_MODEL_PATH, device=None):
    """
    Loads PyTorch model from checkpoint if available, otherwise returns initialized model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Net(num_classes=len(CLASS_NAMES))
    checkpoint_loaded = False
    
    if os.path.exists(checkpoint_path):
        import __main__
        setattr(__main__, 'Net', Net)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint
        checkpoint_loaded = True
        print(f"Loaded model checkpoint from '{checkpoint_path}'")
    else:
        print(f"Warning: Checkpoint '{checkpoint_path}' not found. Using initialized model.")
        
    model.to(device)
    model.eval()
    return model, device, checkpoint_loaded


def predict_reciter(audio_source, checkpoint_path=DEFAULT_MODEL_PATH, offset_sec=5.0):
    """
    Given an audio file path or file stream, extracts features and runs prediction.
    Returns:
        dict containing predicted_class, predicted_name_en, predicted_name_ar,
        confidence, probabilities dict, mel_spectrogram, signal, sample_rate, checkpoint_loaded
    """
    batch_tensor, first_spec, signal, sample_rate = extract_mel_spectrogram(audio_source, offset_sec=offset_sec)

    model, device, checkpoint_loaded = load_model(checkpoint_path)
    batch_tensor = batch_tensor.to(device)

    with torch.no_grad():
        logits = model(batch_tensor)
        probs = F.softmax(logits, dim=1)
        # Average across chunks
        avg_probs = torch.mean(probs, dim=0).cpu().numpy()

    top_idx = int(np.argmax(avg_probs))
    predicted_class = CLASS_NAMES[top_idx]
    confidence = float(avg_probs[top_idx])

    probabilities = {
        CLASS_NAMES[i]: {
            "name_en": RECITERS_INFO[CLASS_NAMES[i]]["en"],
            "name_ar": RECITERS_INFO[CLASS_NAMES[i]]["ar"],
            "probability": float(avg_probs[i])
        }
        for i in range(len(CLASS_NAMES))
    }

    # Sort probabilities descending
    sorted_probs = dict(sorted(probabilities.items(), key=lambda item: item[1]["probability"], reverse=True))

    return {
        "predicted_class": predicted_class,
        "predicted_name_en": RECITERS_INFO[predicted_class]["en"],
        "predicted_name_ar": RECITERS_INFO[predicted_class]["ar"],
        "description": RECITERS_INFO[predicted_class]["desc"],
        "confidence": confidence,
        "probabilities": sorted_probs,
        "spectrogram": first_spec,
        "signal": signal,
        "sample_rate": sample_rate,
        "checkpoint_loaded": checkpoint_loaded,
        "num_chunks": batch_tensor.shape[0]
    }
