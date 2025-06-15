# Audio Classification and Authentication Web App

This project provides a Flask-based web application for audio classification and voice-based authentication, leveraging a ResNet model trained on audio spectrogram images.

## Overview
The application allows users to:

- Upload audio files to classify them.
- Authenticate users based on voice input via a web interface.

## Project Structure


├── app.py                # Flask application setup and routing<br>
├── run.py                # Entry point to run the Flask server<br>
├── login.html            # HTML page for voice-based user authentiation<br>
├── utils<br>
│   ├── audio_utils.py    # Utility functions for audio processing<br>
│   └── model_utils.py    # Utility functions for model initialization and inference
└── models<br>
└── best_model.pt     # Pretrained ResNet model for audio classification

## Dependencies
The project was developed with: [requirements.txt](https://github.com/KajkoIKokosz/PW_CV_PG/blob/main/requirements.txt) with main components:
- Python 3.x
- Flask
- PyTorch
- torchvision
- librosa
- numpy
- Pillow

> To install, simply run:
> ```bash
> pip install -r requirements.txt


## Running the Application

### Start the Server

Run the Flask app:
The app will start at `http://127.0.0.1:5000`.

### Usage

#### Audio Classification

Navigate to the home page (`/`), upload an audio file (supported formats: `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`), and get immediate classification results.

#### Voice Authentication

Navigate to `/login.html`, click on the microphone button, and record your voice for authentication. The app processes the audio and returns whether access is granted or denied.

## Model Details

- **Architecture:** ResNet18 with dropout
- **Input:** Mel spectrogram images derived from audio files
- **Output Classes:** Customize in `app.py` under the `CLASSES` variable.

## Customization

- Update class names and authentication logic in `model_utils.py`.
- Adjust file upload and audio handling in `audio_utils.py`.

## License

This project is open-sourced under the MIT License.
