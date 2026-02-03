from app import app, db, PredictionResult

def verify_db():
    print("Verifying Database...")
    with app.app_context():
        try:
            db.create_all()
            print("Tables ensured.")
            
            # Count existing
            count = PredictionResult.query.count()
            print(f"Current record count: {count}")
            
            # Try adding a dummy record
            test_res = PredictionResult(
                filename="test_db_verify.wav",
                audio_emotion="happy",
                visual_emotion="neutral",
                final_emotion="happy",
                confidence=0.99
            )
            db.session.add(test_res)
            db.session.commit()
            print(f"Test record added. ID: {test_res.id}")
            
            # Verify it's there
            check = PredictionResult.query.get(test_res.id)
            if check:
                print(f"Verification successful: Found record {check}")
                # Clean up
                db.session.delete(check)
                db.session.commit()
                print("Test record verified and cleaned up.")
            else:
                print("Verification FAILED: Record not found after commit.")
                
        except Exception as e:
            print(f"Database Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    verify_db()
