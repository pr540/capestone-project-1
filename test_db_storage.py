from app import app, db, PredictionResult
from datetime import datetime

with app.app_context():
    try:
        new_res = PredictionResult(
            filename="test_manual.wav",
            audio_emotion="happy",
            visual_emotion="N/A",
            final_emotion="happy",
            confidence=0.99
        )
        db.session.add(new_res)
        db.session.commit()
        print("Successfully stored test prediction.")
        
        last = PredictionResult.query.order_by(PredictionResult.id.desc()).first()
        print(f"Stored ID: {last.id}, Filename: {last.filename}")
    except Exception as e:
        print(f"Failed to store: {e}")
        db.session.rollback()
