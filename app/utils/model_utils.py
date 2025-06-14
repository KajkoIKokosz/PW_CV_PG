import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image
from torchvision import transforms

from .audio_utils import load_waveform, waveform_to_spectrogram_pil

def init_model(model_path, classes, device):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def predict_from_file(model, classes, filepath, device):
    # 1) load/convert waveform
    wav, sr = load_waveform(filepath)
    # 2) spectrogram→PIL
    img = waveform_to_spectrogram_pil(wav, sr)
    # 3) to tensor
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    inp = tf(img).unsqueeze(0).to(device)
    # 4) predict
    with torch.no_grad():
        out = model(inp)
        probs = torch.softmax(out, dim=1)
        p, idx = probs.max(1)
    return classes[idx.item()], float(p.item())
