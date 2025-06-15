import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from utils.audio_utils import audio_to_spectrogram_image

# ─── INFERENCE TRANSFORMS ─────────────────────────────────────────────────────
predict_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),       # 1→3 channels
    transforms.Resize((224, 224)),                     # match ResNet input
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),            # same mean/std as training
])

# ─── MODEL DEFINITION ─────────────────────────────────────────────────────────
class ResNetWithDropout(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.base_model = models.resnet18(pretrained=False)
        in_feats = self.base_model.fc.in_features
        # replace fc with (dropout + new Linear)
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_feats, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

# ─── INITIALIZATION & PREDICTION ──────────────────────────────────────────────
def init_model(model_path: str, classes: list, device):
    model = ResNetWithDropout(num_classes=len(classes), dropout_rate=0.5)
    raw_sd = torch.load(model_path, map_location=device)

    # if your checkpoint keys are flat (conv1.weight, bn1.weight, etc),
    # we need to remap them under "base_model.<…>"
    if not any(k.startswith("base_model.") for k in raw_sd):
        new_sd = {}
        for k, v in raw_sd.items():
            if k.startswith("fc."):
                # fc.weight → base_model.fc.1.weight; fc.bias → base_model.fc.1.bias
                suffix = k.split(".", 1)[1]
                new_key = f"base_model.fc.1.{suffix}"
            else:
                # all other layers → base_model.<same name>
                new_key = f"base_model.{k}"
            new_sd[new_key] = v
        state_dict = new_sd
    else:
        # already has "base_model." prefix
        state_dict = raw_sd

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_from_file(model, file_path: str, classes: list, device):
    # 1) audio → spectrogram image
    img = audio_to_spectrogram_image(file_path)
    # 2) apply the same transforms as training
    x = predict_transform(img).unsqueeze(0).to(device)
    # 3) forward pass
    with torch.no_grad():
        out   = model(x)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    # map internal class names to user-facing strings:
    label = {"classA": "access granted", "classB": "access denied"}.get(classes[idx], classes[idx])
    return label, float(probs[idx])
    # return classes[idx], float(probs[idx])
