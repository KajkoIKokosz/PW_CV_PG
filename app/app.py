import os
import uuid
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.model_utils import init_model, predict_from_file, ResNetWithDropout
from utils.audio_utils import save_audio_file, audio_to_spectrogram_image

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "..", "models", "best_model.pt")
TRAIN_DATA_DIR  = os.path.join(BASE_DIR, "training_data")   # will contain granted/denied subfolders

CLASSES         = ["granted", "denied"]
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams for retraining
BATCH_SIZE      = 8
LEARNING_RATE   = 1e-4
NUM_EPOCHS      = 5

# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = Flask(__name__)
model = init_model(MODEL_PATH, CLASSES, DEVICE)

# ─── DATASET FOR FULL RETRAIN ─────────────────────────────────────────────────
class AudioTrainDataset(Dataset):
    def __init__(self, data_dir, classes, transform=None):
        self.samples = []
        for idx, cls in enumerate(classes):
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(".wav"):
                    self.samples.append((os.path.join(cls_dir, fname), idx))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = audio_to_spectrogram_image(path)
        if self.transform:
            img = self.transform(img)
        return img, label

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def retrain_model():
    """Full‐dataset fine‐tune, then overwrite MODEL_PATH and reload global model."""
    dataset = AudioTrainDataset(TRAIN_DATA_DIR, CLASSES, transform=train_transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    net = ResNetWithDropout(num_classes=len(CLASSES), dropout_rate=0.5).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    missing, unexpected = net.load_state_dict(checkpoint, strict=False)
    print(f"[Retrain] Missing keys: {missing}")
    print(f"[Retrain] Unexpected keys: {unexpected}")

    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[Retrain] Epoch {epoch}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

    torch.save(net.state_dict(), MODEL_PATH)
    global model
    model = init_model(MODEL_PATH, CLASSES, DEVICE)

def incremental_update(file_path, label_idx, steps=15, lr=5e-4):
    """
    Fine‐tune the global `model` on one new example so “hard” cases flip correctly.
    """
    global model
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    img = audio_to_spectrogram_image(file_path)
    img = train_transform(img).unsqueeze(0).to(DEVICE)
    target = torch.tensor([label_idx], device=DEVICE)

    for _ in range(steps):
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
    model = init_model(MODEL_PATH, CLASSES, DEVICE)

# ─── ROUTES ──────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return render_template_string("""
    <!doctype html>
    <title>Audio Classifier & Retrainer</title>
    <h1>1) Upload or Record for Prediction</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="audio/*" required>
      <button type="submit">Upload & Predict</button>
    </form>
    <hr>
    <h1>2) Upload New .wav to Retrain</h1>
    <form action="/retrain" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept=".wav" required>
      <select name="label" required>
        <option value="">-- select class --</option>
        <option value="granted">granted</option>
        <option value="denied">denied</option>
      </select>
      <button type="submit">Upload & Retrain</button>
    </form>
    """), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    saved_path = save_audio_file(f)
    label, score = predict_from_file(model, saved_path, CLASSES, DEVICE)
    return jsonify({"label": label, "score": score}), 200

@app.route("/retrain", methods=["POST"])
def retrain_endpoint():
    if "file" not in request.files or "label" not in request.form:
        return jsonify({"error": "Must provide both file and label"}), 400

    f     = request.files["file"]
    label = request.form["label"]
    if label not in CLASSES:
        return jsonify({"error": f"Unknown label: {label}"}), 400
    if not f.filename.lower().endswith(".wav"):
        return jsonify({"error": "Only .wav files accepted for retraining"}), 400

    # Save into training_data/<label>/
    dest_dir = os.path.join(TRAIN_DATA_DIR, label)
    os.makedirs(dest_dir, exist_ok=True)
    filename  = secure_filename(f.filename)
    dest_path = os.path.join(dest_dir, filename)
    f.save(dest_path)

    try:
        # 1) Full‐dataset retrain
        retrain_model()
        # 2) Incremental fine‐tune on this one file
        idx = CLASSES.index(label)
        incremental_update(dest_path, idx)
    except Exception as e:
        return jsonify({"error": f"Retraining failed: {e}"}), 500

    return jsonify({"status": "retrained", "example": filename}), 200

# ─── ENTRYPOINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    for cls in CLASSES:
        os.makedirs(os.path.join(TRAIN_DATA_DIR, cls), exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
