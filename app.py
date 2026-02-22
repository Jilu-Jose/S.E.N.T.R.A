import smtplib
from email.message import EmailMessage
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from database import conn, cursor
import joblib
import json
from threading import Lock
import os
import base64
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load models
emotion_model = load_model(
    "models/fer2013_mini_XCEPTION.102-0.66.hdf5",
    compile=False
)

task_bundle = joblib.load("models/task_recommender_rf.pkl")
task_model = task_bundle["model"]
mood_encoder = task_bundle["mood_encoder"]
task_encoder = task_bundle["task_encoder"]

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

emotion_to_mood = {
    "Happy": "Happy",
    "Neutral": "Neutral",
    "Sad": "Sad",
    "Surprise": "Surprise",
    "Angry": "Sad",
    "Fear": "Sad",
    "Disgust": "Sad"
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

stress_emotions = ["Sad", "Angry", "Fear", "Disgust"]

 
current_emotion = "Neutral"
current_task = "No task assigned"
stress_count = 0
alert_active = False
lock = Lock()


def send_hr_alert(employee_id, emotion):
    msg = EmailMessage()
    msg.set_content(
        f"""
        Stress Alert Generated

        Employee ID: {employee_id}
        Detected Emotion: {emotion}

        Prolonged stress has been detected.
        Please take appropriate action.
        """
    )

    EMAIL_ID = os.getenv("EMAIL_ID")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    HR_EMAIL = os.getenv("HR_EMAIL")

    msg["Subject"] = "WARNING Stress Alert – S.E.N.T.R.A System: Employee needs assistance!!"
    msg["From"] = EMAIL_ID # EMAIL_ID
    msg["To"] = HR_EMAIL # HR_EMAIL

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ID, EMAIL_PASSWORD) # EMAIL_PASSWORD
            smtp.send_message(msg)
    except Exception as e:
        print(f"Email error: {e}")


@app.route('/process_frame', methods=['POST'])
def process_frame():
    global current_emotion, current_task, stress_count, alert_active
    
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data'}), 400
            
        image_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_detected = False
        
        for (x, y, w, h) in faces:
            face_detected = True
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            face = face / 255.0
            face = face.reshape(1, 64, 64, 1)
            
            preds = emotion_model.predict(face, verbose=0)
            emotion = emotion_labels[np.argmax(preds)]
            
            mapped_mood = emotion_to_mood[emotion]
            mood_encoded = mood_encoder.transform([mapped_mood])[0]
            
            current_workload = np.random.randint(3, 8)
            sleeping_hours = np.random.randint(5, 9)
            working_hours = np.random.randint(6, 10)
            deadline_pressure = np.random.randint(2, 8)

            task_input = pd.DataFrame([{
                "Mood": mood_encoded,
                "Current_Workload": current_workload,
                "Sleeping_Hours": sleeping_hours,
                "Working_Hours": working_hours,
                "Deadline_Pressure": deadline_pressure
            }])
            
            task_encoded = task_model.predict(task_input)
            recommended_task = task_encoder.inverse_transform(task_encoded)[0]
            
            with lock:
                current_emotion = emotion
                current_task = recommended_task
                
                cursor.execute(
                    "INSERT INTO mood_logs VALUES (NULL, ?, ?, ?, ?)",
                    (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "EMP001",
                        emotion,
                        recommended_task
                    )
                )
                conn.commit()
                
                if emotion in stress_emotions:
                    stress_count += 1
                else:
                    stress_count = 0
                
                if stress_count >= 5:
                    alert_active = True
                    send_hr_alert("EMP001", emotion)
                    stress_count = 0
                else:
                    alert_active = False
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame, emotion, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )
        
        _, buffer = cv2.imencode('.jpg', frame)
        processed_img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'image': 'data:image/jpeg;base64,' + processed_img_base64,
            'emotion': current_emotion if face_detected else "Neutral",
            'task': current_task if face_detected else "No task assigned"
        })
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def landing():
    return render_template("landing.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/app')
def app_route():
    return render_template("index.html")


@app.route('/status')
def status():
    with lock:
        return jsonify({
            'emotion': current_emotion,
            'task': current_task,
            'alert': alert_active
        })


@app.route('/api/history')
def get_history():
    data = cursor.execute(
        "SELECT * FROM mood_logs ORDER BY timestamp DESC LIMIT 100"
    ).fetchall()
    
    history = []
    for row in data:
        history.append({
            'id': row[0],
            'timestamp': row[1],
            'employee_id': row[2],
            'emotion': row[3],
            'task': row[4]
        })
    
    return jsonify(history)


@app.route('/api/recent_alerts')
def get_recent_alerts():
    data = cursor.execute(
        "SELECT * FROM mood_logs WHERE emotion IN ('Sad', 'Angry', 'Fear', 'Disgust') ORDER BY timestamp DESC LIMIT 5"
    ).fetchall()
    
    alerts = []
    for row in data:
        alerts.append({
            'id': row[0],
            'timestamp': row[1],
            'employee_id': row[2],
            'emotion': row[3],
            'task': row[4]
        })
    
    return jsonify(alerts)


@app.route('/api/analytics')
def get_analytics():
  
    emotion_data = cursor.execute(
        "SELECT emotion, COUNT(*) as count FROM mood_logs GROUP BY emotion"
    ).fetchall()
    
    emotions = {}
    for row in emotion_data:
        emotions[row[0]] = row[1]
    
  
    recent_data = cursor.execute(
        """SELECT emotion, COUNT(*) as count 
           FROM mood_logs 
           WHERE timestamp >= datetime('now', '-1 day')
           GROUP BY emotion"""
    ).fetchall()
    
    recent_emotions = {}
    for row in recent_data:
        recent_emotions[row[0]] = row[1]
    
    
    hourly_data = cursor.execute(
        """SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
           FROM mood_logs
           WHERE timestamp >= datetime('now', '-1 day')
           GROUP BY hour
           ORDER BY hour"""
    ).fetchall()
    
    hourly = {}
    for row in hourly_data:
        hourly[row[0]] = row[1]
    
    return jsonify({
        'emotion_distribution': emotions,
        'recent_emotions': recent_emotions,
        'hourly_distribution': hourly
    })



if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=10000)