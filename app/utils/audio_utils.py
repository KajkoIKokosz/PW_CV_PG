import os
import torchaudio
from pydub import AudioSegment

def load_waveform(path):
    base, ext = os.path.splitext(path)
    ext = ext.lower().lstrip('.')
    if ext != 'wav':
        sound = AudioSegment.from_file(path, format=ext)
        wav_path = base + '__conv.wav'
        sound.export(wav_path, format='wav')
        path = wav_path

    waveform, sr = torchaudio.load(path)
    if ext != 'wav' and os.path.exists(wav_path):
        os.remove(wav_path)
    return waveform, sr

def waveform_to_spectrogram_pil(waveform, sr):
    from torchvision import transforms
    from PIL import Image
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=128, n_fft=1024, hop_length=512
    )(waveform)
    melspec_db = torchaudio.transforms.AmplitudeToDB()(melspec)
    norm = (melspec_db - melspec_db.min()) / (melspec_db.max() - melspec_db.min())
    arr = (norm * 255).transpose(1,2).squeeze(0).byte().cpu().numpy()
    return Image.fromarray(arr, mode='L').convert('RGB').resize((224,224))
