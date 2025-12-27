from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class PredictionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    audio_emotion = db.Column(db.String(50))
    visual_emotion = db.Column(db.String(50))
    final_emotion = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    
    def __repr__(self):
        return f'<Prediction {self.id} - {self.final_emotion}>'
