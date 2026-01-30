import sqlite3

conn = sqlite3.connect("mood_data.db")
cursor = conn.cursor()

data = cursor.execute("SELECT * FROM mood_logs").fetchall()

for row in data:
    print(row)










