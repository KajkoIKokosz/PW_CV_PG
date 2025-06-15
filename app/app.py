import os
import uuid
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
from pydub import AudioSegment

from utils.model_utils import init_model, predict_from_file
from utils.audio_utils import save_audio_file

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_model.pt")

# Replace these with the exact class‐names & order you used in training:
CLASSES = ["classA", "classB"]  # e.g. ["access granted","access denied"]

DEVICE = "cuda" if os.getenv("USE_CUDA", "0") == "1" else "cpu"

# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = Flask(__name__)
model = init_model(MODEL_PATH, CLASSES, DEVICE)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def convert_to_wav(src_path: str) -> str:
    """
    Take any audio file (webm/ogg/mp3/etc) and output a .wav file path.
    """
    base, _ = os.path.splitext(src_path)
    wav_path = f"{base}.wav"
    audio = AudioSegment.from_file(src_path)
    audio.export(wav_path, format="wav")
    return wav_path

# ─── ROUTES ──────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return render_template_string("""
    <!doctype html>
    <html>
    <head>
      <title>Audio Classifier</title>
    </head>
    <body>
      <h1>Upload or Record Audio for Prediction</h1>

      <form id="uploadForm" enctype="multipart/form-data">
        <input type="file" name="file" accept="audio/*">
        <button type="submit">Upload & Predict</button>
      </form>

      <p>— or —</p>

      <button id="recordBtn">🎤 Record (3s) & Predict</button>

      <p id="result"></p>

      <script>
        const uploadForm = document.getElementById('uploadForm');
        const recordBtn = document.getElementById('recordBtn');
        const result    = document.getElementById('result');

        uploadForm.onsubmit = async (e) => {
          e.preventDefault();
          result.textContent = '⏳ Processing upload…';
          const fd = new FormData(uploadForm);
          const resp = await fetch('/predict', { method:'POST', body: fd });
          const js   = await resp.json();
          if (resp.ok) {
            result.textContent = `Label: ${js.label} (score: ${js.score.toFixed(2)})`;
          } else {
            result.textContent = `❌ Error: ${js.error||'Unknown'}`;
          }
        };

        recordBtn.onclick = async () => {
          recordBtn.disabled   = true;
          result.textContent   = '🔴 Recording…';
          const stream         = await navigator.mediaDevices.getUserMedia({ audio: true });
          const recorder       = new MediaRecorder(stream);
          let chunks           = [];
          recorder.ondataavailable = e => chunks.push(e.data);
          recorder.start();
          setTimeout(() => recorder.stop(), 3000);
          recorder.onstop = async () => {
            result.textContent = '⏳ Uploading…';
            const blob = new Blob(chunks, { type: recorder.mimeType });
            const fd   = new FormData();
            fd.append('file', blob, 'recording.webm');
            const resp = await fetch('/predict', { method:'POST', body: fd });
            const js   = await resp.json();
            if (resp.ok) {
              result.textContent = `Label: ${js.label} (score: ${js.score.toFixed(2)})`;
            } else {
              result.textContent = `❌ Error: ${js.error||'Unknown'}`;
            }
            recordBtn.disabled = false;
          };
        };
      </script>
    </body>
    </html>
    """), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # 1) Save raw upload
    saved_path = save_audio_file(f)

    # 2) If not a true WAV on disk, convert to WAV
    if not saved_path.lower().endswith(".wav"):
        try:
            saved_path = convert_to_wav(saved_path)
        except Exception as e:
            return jsonify({"error": f"Audio conversion failed: {str(e)}"}), 500

    # 3) Run inference (audio → spectrogram → model)
    try:
        label, score = predict_from_file(model, saved_path, CLASSES, DEVICE)
        return jsonify({"label": label, "score": score}), 200
    except Exception as e:
        return jsonify({"error": f"Inference error: {str(e)}"}), 500

# ─── ENTRYPOINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
