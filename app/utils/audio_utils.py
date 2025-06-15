import os
import uuid
import librosa
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}

def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def save_audio_file(file_storage, upload_folder="uploads"):
    """
    Save the incoming audio FileStorage to disk and return its path.
    """
    orig = secure_filename(file_storage.filename or "")
    ext  = orig.rsplit('.', 1)[-1].lower()
    if orig and ext in ALLOWED_EXTENSIONS:
        filename = orig
    else:
        filename = f"{uuid.uuid4().hex}.wav"
    os.makedirs(upload_folder, exist_ok=True)
    path = os.path.join(upload_folder, filename)
    file_storage.save(path)
    return path

def audio_to_spectrogram_image(file_path, sr=22050, n_mels=128, fmax=None):
    """
    Load audio, compute a log-scaled mel spectrogram,
    and return a PIL Image (uint8) suitable for the ResNet transforms.
    """
    y, sr = librosa.load(file_path, sr=sr)
    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    S_db = librosa.power_to_db(S, ref=np.max)
    # scale to 0–255
    img = (255 * (S_db - S_db.min()) / (S_db.max() - S_db.min())).astype(np.uint8)
    return Image.fromarray(img)
