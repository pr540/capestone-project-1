import sqlite3
import os

db_path = os.path.join('instance', 'emotions.db')
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(prediction_result)")
    columns = cursor.fetchall()
    print("Columns in prediction_result:")
    for col in columns:
        print(col)
    conn.close()
else:
    print("DB file not found.")
