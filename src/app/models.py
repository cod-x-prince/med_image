from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(50), default='doctor') # 'admin' or 'doctor'

class Upload(db.Model):
    __tablename__ = 'uploads'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256), nullable=False)
    prediction = db.Column(db.Text, nullable=True) # Storing JSON as text
    report = db.Column(db.Text, nullable=True)
    date = db.Column(db.String(50), default=datetime.now().strftime("%Y-%m-%d %H:%M"))
    patient_id = db.Column(db.String(50), index=True, nullable=True)
    patient_name = db.Column(db.String(100), nullable=True)
    patient_sex = db.Column(db.String(10), nullable=True)
    patient_age = db.Column(db.String(10), nullable=True)
    
    # Smart Deployment Fields
    scan_type = db.Column(db.String(20), default='standard') # standard, deep, ensemble
    model_used = db.Column(db.String(100), nullable=True)
    inference_time = db.Column(db.Float, nullable=True) 
    auc_score = db.Column(db.Float, nullable=True)

class AuditLog(db.Model):
    __tablename__ = 'audit_logs'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    action = db.Column(db.String(256), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    details = db.Column(db.Text, nullable=True)
