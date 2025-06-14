import os
import torch
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from utils.model_utils import init_model, predict_from_file

# --- config ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = ['green', 'red']                                   # your two classes
MODEL_PATH = '/home/dog4/PhD/computer_vision/cvproj/best_model.pt'
RECORD_FOLDER = os.path.join(os.path.dirname(__file__), 'recordings')
os.makedirs(RECORD_FOLDER, exist_ok=True)

# --- app setup ---
app = Flask(__name__, template_folder='views', static_folder='static')
CORS(app)

# load once
model = init_model(MODEL_PATH, CLASSES, DEVICE)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'audio' not in request.files:
        return jsonify({'status':'error','error':'no audio'})
    f = request.files['audio']
    save_path = os.path.join(RECORD_FOLDER, f.filename)
    f.save(save_path)

    user, score = predict_from_file(model, CLASSES, save_path, DEVICE)
    status = 'granted' if score >= 0.6 else 'denied'
    return jsonify({'status': status, 'user': user, 'score': score})

