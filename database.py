import sqlite3

conn = sqlite3.connect("mood_data.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS mood_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    employee_id TEXT,
    emotion TEXT,
    recommended_task TEXT
)
""")

conn.commit()