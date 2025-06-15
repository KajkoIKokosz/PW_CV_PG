import os
import uuid
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.model_utils import init_model, predict_from_file, ResNetWithDropout
from utils.audio_utils import save_audio_file, audio_to_spectrogram_image

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, os.pardir, "models", "best_model.pt")
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "training_data")
CLASSES        = ["granted", "denied"]
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters for retraining
BATCH_SIZE     = 8
LEARNING_RATE  = 1e-4
NUM_EPOCHS     = 5

# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="views")
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

# Match inference preprocessing
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def retrain_model():
    """
    Performs full-dataset fine-tuning and reloads the global model.
    """
    dataset = AudioTrainDataset(TRAIN_DATA_DIR, CLASSES, transform=train_transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    net = ResNetWithDropout(num_classes=len(CLASSES), dropout_rate=0.5).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    net.load_state_dict(checkpoint, strict=False)

    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS+1):
        total_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg = total_loss / len(dataset)
        yield_log = f"Epoch {epoch}/{NUM_EPOCHS}   Loss: {avg:.4f}\n"
        print(yield_log, end="")
        yield yield_log

    # Reload global model
    torch.save(net.state_dict(), MODEL_PATH)
    global model
    model = init_model(MODEL_PATH, CLASSES, DEVICE)


def incremental_update(file_path, label_idx, steps=15, lr=5e-4):
    """
    Extra fine-tuning on the single new example for real-time streaming.
    """
    global model
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for step in range(1, steps+1):
        img = audio_to_spectrogram_image(file_path)
        img = train_transform(img).unsqueeze(0).to(DEVICE)
        target = torch.tensor([label_idx], device=DEVICE)

        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        step_log = f"Step {step}/{steps}   Loss: {loss.item():.4f}\n"
        print(step_log, end="")
        yield step_log

    # Final save & reload
    torch.save(model.state_dict(), MODEL_PATH)
    model = init_model(MODEL_PATH, CLASSES, DEVICE)

# ─── ROUTES ──────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/login", methods=["GET"])
def login():
    return render_template("login.html")

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
    # Validation & save
    if "file" not in request.files or "label" not in request.form:
        return jsonify({"error": "Must provide both file and label"}), 400
    f = request.files["file"]
    label = request.form["label"]
    if label not in CLASSES:
        return jsonify({"error": f"Unknown label: {label}"}), 400
    if not f.filename.lower().endswith(".wav"):
        return jsonify({"error": "Only .wav files accepted for retraining"}), 400

    dest_dir = os.path.join(TRAIN_DATA_DIR, label)
    os.makedirs(dest_dir, exist_ok=True)
    filename = secure_filename(f.filename)
    dest_path = os.path.join(dest_dir, filename)
    f.save(dest_path)

    # Stream training logs
    def generate_logs():
        yield "Starting full retrain...\n"
        for line in retrain_model():
            yield line
        yield "\nStarting incremental update...\n"
        for line in incremental_update(dest_path, CLASSES.index(label)):
            yield line
        yield "\n✅ Retraining complete!\n"

    return Response(generate_logs(), mimetype="text/plain; charset=utf-8")

# ─── ENTRYPOINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    for cls in CLASSES:
        os.makedirs(os.path.join(TRAIN_DATA_DIR, cls), exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)