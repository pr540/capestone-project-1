from app import app, db, PredictionResult

with app.app_context():
    # Query all results
    results = PredictionResult.query.all()
    print(f"Total Predictions in DB: {len(results)}")
    
    # Show the last 5
    print("\nLast 5 Predictions:")
    for r in results[-5:]:
        print(f"ID: {r.id} | File: {r.filename} | Audio: {r.audio_emotion} | Visual: {r.visual_emotion} | Final: {r.final_emotion} | Conf: {r.confidence}")
