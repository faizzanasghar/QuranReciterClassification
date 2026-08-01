import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import pickle
from skimage.transform import resize

# English & Arabic metadata for the 12 reciters
RECITERS_INFO = {
    "AbdulBari_Althubaity": {
        "en": "Abdul Bari Al-Thubaity",
        "ar": "الشيخ عبد الباري الثبيتي",
        "desc": "Imam and Khateeb of Al-Masjid an-Nabawi in Madinah."
    },
    "AbdulRahman_Alsudais": {
        "en": "Abdul Rahman Al-Sudais",
        "ar": "الشيخ عبد الرحمن السديس",
        "desc": "General President for the Affairs of the Two Holy Mosques & Imam of Masjid al-Haram."
    },
    "Abdullah_Albuaijan": {
        "en": "Abdullah Al-Buaijan",
        "ar": "الشيخ عبد الله البعيجان",
        "desc": "Imam and Khateeb of Al-Masjid an-Nabawi in Madinah."
    },
    "Ali_Alhothaify": {
        "en": "Ali Al-Huthaify",
        "ar": "الشيخ علي عبد الرحمن الحذيفي",
        "desc": "Chief Imam of Al-Masjid an-Nabawi in Madinah."
    },
    "Bander_Balilah": {
        "en": "Bandar Baleela",
        "ar": "الشيخ بندر بليلة",
        "desc": "Imam and Khateeb of Masjid al-Haram in Makkah."
    },
    "Maher_Almuaiqly": {
        "en": "Maher Al-Muaiqly",
        "ar": "الشيخ ماهر المعيقلي",
        "desc": "Renowned Imam of Masjid al-Haram in Makkah."
    },
    "Mohammed_Aluhaidan": {
        "en": "Muhammad Al-Luhaidan",
        "ar": "الشيخ محمد اللحيدان",
        "desc": "Prominent Qari and Judge from Saudi Arabia."
    },
    "Mohammed_Ayoub": {
        "en": "Muhammad Ayyub",
        "ar": "الشيخ محمد أيوب",
        "desc": "Late former Imam of Al-Masjid an-Nabawi."
    },
    "Nasser_Alqutami": {
        "en": "Nasser Al-Qatami",
        "ar": "الشيخ ناصر القطامي",
        "desc": "Famous Qari and Imam of Abdullah Bin Nasser Al-Muhaini Mosque in Riyadh."
    },
    "Saad_Alghamdi": {
        "en": "Saad Al-Ghamdi",
        "ar": "الشيخ سعد الغامدي",
        "desc": "World-renowned Qari and former Imam of Masjid an-Nabawi."
    },
    "Saud_Alshuraim": {
        "en": "Saud Al-Shuraim",
        "ar": "الشيخ سعود الشريم",
        "desc": "Former long-time Imam & Khateeb of Masjid al-Haram in Makkah."
    },
    "Yasser_Aldossary": {
        "en": "Yasser Al-Dosari",
        "ar": "الشيخ ياسر الدوسري",
        "desc": "Imam and Khateeb of Masjid al-Haram in Makkah."
    }
}

# Alphabetically sorted classes corresponding to LabelEncoder
CLASS_NAMES = sorted(list(RECITERS_INFO.keys()))


class Net(nn.Module):
    """
    CNN Architecture for Quran Audio Reciter Classification.
    """
    def __init__(self, num_classes=12):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64 * 16 * 32, 1024)
        self.linear2 = nn.Linear(1024, 256)
        self.output = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pooling(x)
        x = self.relu(self.conv2(x))
        x = self.pooling(x)
        x = self.relu(self.conv3(x))
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        x = self.output(x)
        return x


class QuranAudioClassifier:
    """
    Helper wrapper to preprocess input audio files and run predictions.
    """
    def __init__(self, model, label_encoder=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.label_encoder = label_encoder

    def extract_spectrogram(self, file_path, sr=22050, duration=5, img_height=128, img_width=256):
        """
        Loads an audio file and converts it into a Mel Spectrogram image tensor.
        """
        signal, sr = librosa.load(file_path, sr=sr, duration=duration)
        spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        
        # Pad/truncate length to match training shape
        spec_resized = librosa.util.fix_length(spec_db, size=(duration * sr) // 512 + 1)
        spec_resized = resize(spec_resized, (img_height, img_width), anti_aliasing=True)
        
        # Format tensor shape for model input: (Batch_Size=1, Channels=1, Height, Width)
        tensor = torch.tensor(spec_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor

    def predict(self, audio_path):
        """
        Predicts reciter class for a given audio file path.
        Returns predicted class index, reciter name (if encoder provided), and confidence.
        """
        tensor = self.extract_spectrogram(audio_path).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            
            idx = predicted_idx.item()
            conf = confidence.item()

        reciter_name = None
        if self.label_encoder is not None:
            reciter_name = self.label_encoder.inverse_transform([idx])[0]

        return {
            "class_index": idx,
            "reciter_name": reciter_name,
            "confidence": conf
        }


def load_classifier(model_path, num_classes=12):
    """
    Factory function to load model weights/bundle (.pkl, .pth, or .pt).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_path.endswith('.pkl'):
        # Loaded from quran_classifier_bundle.pkl
        with open(model_path, 'rb') as f:
            bundle = pickle.load(f)
            
        model = Net(num_classes=bundle.get('num_classes', num_classes))
        model.load_state_dict(bundle['model_state_dict'])
        label_encoder = bundle.get('label_encoder')
        return QuranAudioClassifier(model=model, label_encoder=label_encoder, device=device)

    elif model_path.endswith('.pth'):
        # Loaded from model_weights.pth
        model = Net(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return QuranAudioClassifier(model=model, device=device)

    elif model_path.endswith('.pt'):
        # Loaded full model or checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, Net):
            model = checkpoint
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model = Net(num_classes=num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model = Net(num_classes=num_classes)
            model.load_state_dict(checkpoint)
            
        return QuranAudioClassifier(model=model, device=device)
    else:
        raise ValueError("Unsupported file extension. Use .pkl, .pth, or .pt")
