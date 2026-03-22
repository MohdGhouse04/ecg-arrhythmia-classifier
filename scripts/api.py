#!/usr/bin/env python3
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.io import loadmat

app = Flask(__name__)
CORS(app)
model, le = None, None

CLASS_INFO = {
    '426177001': {'name':'Sinus Bradycardia','desc':'Heart rate below 60 bpm','icon':'💙','severity':'Low'},
    '426783006': {'name':'Normal Sinus Rhythm','desc':'Normal heart rhythm','icon':'💚','severity':'Normal'},
    '427084000': {'name':'Sinus Tachycardia','desc':'Heart rate above 100 bpm','icon':'🔴','severity':'Moderate'},
    '426761007': {'name':'Supraventricular Tachycardia','desc':'Rapid heart rate from above ventricles','icon':'⚡','severity':'High'},
    '164890007': {'name':'Right Bundle Branch Block','desc':'Delay in right ventricle electrical pathway','icon':'⚠️','severity':'Moderate'},
}

def load_model():
    global model, le
    import tensorflow as tf
    print("Loading model...")
    model = tf.keras.models.load_model("outputs/final_ecg_model.keras", compile=False)
    with open("outputs/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    print(f"Model loaded | Classes: {list(le.classes_)}")

def preprocess(signal):
    signal = signal[::5, :].astype("float32")
    mean = np.mean(signal, axis=0, keepdims=True)
    std  = np.std(signal,  axis=0, keepdims=True)
    return ((signal - mean) / (std + 1e-8))[np.newaxis, ...]

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename.endswith('.mat'):
        return jsonify({'error': 'Only .mat files supported'}), 400
    try:
        tmp = f"/tmp/{file.filename}"
        file.save(tmp)
        mat = loadmat(tmp)
        signal = mat["val"].T
        X = preprocess(signal)
        probs = model.predict(X, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_code = le.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])
        all_probs = {code: round(float(probs[i])*100,1) for i, code in enumerate(le.classes_)}
        info = CLASS_INFO.get(pred_code, {'name':pred_code,'desc':'Unknown','icon':'🫀','severity':'Unknown'})
        return jsonify({'prediction':pred_code,'name':info['name'],'description':info['desc'],
            'icon':info['icon'],'severity':info['severity'],'confidence':round(confidence*100,1),
            'all_probs':all_probs,'filename':file.filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    print("\n========================================")
    print("  ECG Classifier API — http://localhost:5000")
    print("========================================\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
