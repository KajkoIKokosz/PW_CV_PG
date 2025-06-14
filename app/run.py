# run.py
from app import app

if __name__ == "__main__":
    # you can tweak host/port as you like
    app.run(host="127.0.0.1", port=5000, debug=True)