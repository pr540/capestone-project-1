from app import app
from database import db, PredictionResult
import os

with app.app_context():
    count = PredictionResult.query.count()
    print(f"Total Records: {count}")
    last_5 = PredictionResult.query.order_by(PredictionResult.timestamp.desc()).limit(5).all()
    for p in last_5:
        print(f"ID: {p.id}, Emotion: {p.final_emotion}, File: {p.filename}")

db_file = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
print(f"Database File: {db_file}")
print(f"Exists: {os.path.exists(db_file)}")
if os.path.exists(db_file):
    print(f"Size: {os.path.getsize(db_file)} bytes")
