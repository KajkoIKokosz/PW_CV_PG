import os
from flask import Flask, request, jsonify, render_template_string
from utils.model_utils import init_model, predict_from_file
from utils.audio_utils import save_audio_file

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_model.pt")

# Replace these with the exact class‐names & order you used in training:
CLASSES = ["classA", "classB"]  # e.g. ["negative","positive"]

DEVICE = "cuda" if os.getenv("USE_CUDA", "0") == "1" else "cpu"

# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = Flask(__name__)
model = init_model(MODEL_PATH, CLASSES, DEVICE)

# ─── ROUTES ──────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return render_template_string("""
    <!doctype html>
    <title>Audio Classifier</title>
    <h1>Upload an audio file for prediction</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="audio/*" required>
      <button type="submit">Upload & Predict</button>
    </form>
    """), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # save the raw audio
    saved_path = save_audio_file(f)

    # run inference (audio → spectrogram → model)
    label, score = predict_from_file(model, saved_path, CLASSES, DEVICE)
    return jsonify({"label": label, "score": score}), 200

# ─── ENTRYPOINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
