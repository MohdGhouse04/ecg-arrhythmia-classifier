#!/usr/bin/env python3
"""
backend/app.py — Main Flask application
Run: python backend/app.py
"""

import os, sys, pickle, json, numpy as np
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from scipy.io import loadmat

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ecg_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'ecg-secret-key-2024'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    def to_dict(self):
        return {'id':self.id,'name':self.name,'email':self.email,'created_at':self.created_at.strftime('%Y-%m-%d %H:%M:%S'),'total_predictions':len(self.predictions)}

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(200))
    patient_name = db.Column(db.String(100))
    patient_age = db.Column(db.String(10))
    patient_gender = db.Column(db.String(10))
    diagnosis_code = db.Column(db.String(50))
    diagnosis_name = db.Column(db.String(200))
    confidence = db.Column(db.Float)
    severity = db.Column(db.String(50))
    all_probs = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    def to_dict(self):
        return {'id':self.id,'filename':self.filename,'patient_name':self.patient_name,'patient_age':self.patient_age,'patient_gender':self.patient_gender,'diagnosis_code':self.diagnosis_code,'diagnosis_name':self.diagnosis_name,'confidence':self.confidence,'severity':self.severity,'all_probs':json.loads(self.all_probs) if self.all_probs else {},'created_at':self.created_at.strftime('%Y-%m-%d %H:%M:%S')}

model, le = None, None
CLASS_INFO = {
    '426177001':{'name':'Sinus Bradycardia','desc':'Heart rate below 60 bpm','icon':'💙','severity':'Low'},
    '426783006':{'name':'Normal Sinus Rhythm','desc':'Normal heart rhythm','icon':'💚','severity':'Normal'},
    '427084000':{'name':'Sinus Tachycardia','desc':'Heart rate above 100 bpm','icon':'🔴','severity':'Moderate'},
    '426761007':{'name':'Supraventricular Tachycardia','desc':'Rapid heart rate from above ventricles','icon':'⚡','severity':'High'},
    '164890007':{'name':'Right Bundle Branch Block','desc':'Delay in right ventricle electrical pathway','icon':'⚠️','severity':'Moderate'},
}

def load_ml_model():
    global model, le
    import tensorflow as tf
    print("Loading ECG model...")
    model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "final_ecg_model.keras"), compile=False)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "label_encoder.pkl"),"rb") as f: le = pickle.load(f)
    print(f"Model loaded | Classes: {list(le.classes_)}")

def preprocess(signal):
    signal = signal[::5,:].astype("float32")
    return ((signal-np.mean(signal,axis=0,keepdims=True))/(np.std(signal,axis=0,keepdims=True)+1e-8))[np.newaxis,...]

@app.route('/api/register', methods=['POST'])
def register():
    d = request.get_json()
    name,email,password = d.get('name','').strip(),d.get('email','').strip().lower(),d.get('password','')
    if not name or not email or not password: return jsonify({'error':'All fields required'}),400
    if len(password)<6: return jsonify({'error':'Password must be at least 6 characters'}),400
    if User.query.filter_by(email=email).first(): return jsonify({'error':'Email already registered'}),409
    user = User(name=name,email=email,password=bcrypt.generate_password_hash(password).decode('utf-8'))
    db.session.add(user); db.session.commit()
    token = create_access_token(identity=str(user.id))
    return jsonify({'message':'Account created!','token':token,'user':user.to_dict()}),201

@app.route('/api/login', methods=['POST'])
def login():
    d = request.get_json()
    user = User.query.filter_by(email=d.get('email','').strip().lower()).first()
    if not user or not bcrypt.check_password_hash(user.password,d.get('password','')): return jsonify({'error':'Invalid email or password'}),401
    return jsonify({'message':'Login successful','token':create_access_token(identity=str(user.id)),'user':user.to_dict()})

@app.route('/api/me', methods=['GET'])
@jwt_required()
def me():
    user = User.query.get(int(get_jwt_identity()))
    return jsonify({'user':user.to_dict()}) if user else (jsonify({'error':'Not found'}),404)

@app.route('/api/predict', methods=['POST'])
@jwt_required()
def predict():
    if model is None: return jsonify({'error':'Model not loaded'}),500
    file = request.files.get('file')
    if not file or not file.filename.endswith('.mat'): return jsonify({'error':'Upload a valid .mat file'}),400
    try:
        tmp=f"/tmp/{file.filename}"; file.save(tmp)
        signal=loadmat(tmp)["val"].T; X=preprocess(signal)
        probs=model.predict(X,verbose=0)[0]; pred_idx=int(np.argmax(probs))
        pred_code=le.inverse_transform([pred_idx])[0]; confidence=float(probs[pred_idx])
        all_probs={code:round(float(probs[i])*100,1) for i,code in enumerate(le.classes_)}
        info=CLASS_INFO.get(pred_code,{'name':pred_code,'desc':'Unknown','icon':'🫀','severity':'Unknown'})
        p=Prediction(user_id=int(get_jwt_identity()),filename=file.filename,
            patient_name=request.form.get('patient_name','Unknown'),
            patient_age=request.form.get('patient_age',''),
            patient_gender=request.form.get('patient_gender',''),
            diagnosis_code=pred_code,diagnosis_name=info['name'],
            confidence=round(confidence*100,1),severity=info['severity'],
            all_probs=json.dumps(all_probs))
        db.session.add(p); db.session.commit()
        return jsonify({'prediction_id':p.id,'prediction':pred_code,'name':info['name'],'description':info['desc'],'icon':info['icon'],'severity':info['severity'],'confidence':round(confidence*100,1),'all_probs':all_probs,'filename':file.filename,'patient_name':request.form.get('patient_name','Unknown'),'saved':True})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/history', methods=['GET'])
@jwt_required()
def history():
    preds=Prediction.query.filter_by(user_id=int(get_jwt_identity())).order_by(Prediction.created_at.desc()).all()
    return jsonify({'predictions':[p.to_dict() for p in preds],'total':len(preds)})

@app.route('/api/history/<int:pid>', methods=['DELETE'])
@jwt_required()
def delete_pred(pid):
    p=Prediction.query.filter_by(id=pid,user_id=int(get_jwt_identity())).first()
    if not p: return jsonify({'error':'Not found'}),404
    db.session.delete(p); db.session.commit()
    return jsonify({'message':'Deleted'})

@app.route('/api/stats', methods=['GET'])
@jwt_required()
def stats():
    preds=Prediction.query.filter_by(user_id=int(get_jwt_identity())).all()
    total=len(preds)
    if total==0: return jsonify({'total':0,'by_diagnosis':{},'avg_confidence':0})
    by_diag={}; tc=0
    for p in preds: by_diag[p.diagnosis_name]=by_diag.get(p.diagnosis_name,0)+1; tc+=p.confidence
    return jsonify({'total':total,'by_diagnosis':by_diag,'avg_confidence':round(tc/total,1)})

@app.route('/api/health')
def health(): return jsonify({'status':'ok','model_loaded':model is not None})

if __name__=='__main__':
    with app.app_context(): db.create_all(); print("Database ready")
    load_ml_model()
    print("\n========================================")
    print("  ECG Backend — http://localhost:5000")
    print("========================================\n")
    app.run(host='0.0.0.0',port=5000,debug=False)
