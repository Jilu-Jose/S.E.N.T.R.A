
import smtplib
from email.message import EmailMessage
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from database import conn, cursor
import joblib

st.set_page_config(page_title="S.E.N.T.R.A System", layout="wide")


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

if "stress_count" not in st.session_state:
    st.session_state.stress_count = 0

menu = st.sidebar.selectbox(
    "Dashboard Menu",
    ["Emotion Detection", "Mood History"]
)

st.title("S.E.N.T.R.A System")




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

    msg["Subject"] = "🚨 Stress Alert – S.E.N.T.R.A System: Employee needs assistance!!"
    msg["From"] = "jilujose786@gmail.com"
    msg["To"] = "jilupjose111@gmail.com"

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login("jilujose786@gmail.com", "tyik wnen qwal bsyl")
        smtp.send_message(msg)



if menu == "Emotion Detection":

    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    emotion_placeholder = st.empty()
    task_placeholder = st.empty()
    alert_placeholder = st.empty()

    cap = None

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
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

             
                emotion_placeholder.markdown(
                    f"### Detected Emotion: **{emotion}**"
                )
                task_placeholder.success(
                    f"Recommended Task: {recommended_task}"
                )

                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(
                    frame, emotion, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2
                )

                
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
                    st.session_state.stress_count += 1
                else:
                    st.session_state.stress_count = 0

                if st.session_state.stress_count >= 5:
                    alert_placeholder.error("Stress Alert: HR Notified")
                    send_hr_alert("EMP001", emotion)
                    st.session_state.stress_count = 0

                else:
                    alert_placeholder.empty()

            FRAME_WINDOW.image(frame, channels="BGR")

    else:
        if cap:
            cap.release()

elif menu == "Mood History":

    st.subheader("Employee Mood History")

    data = cursor.execute(
        "SELECT * FROM mood_logs ORDER BY timestamp DESC"
    ).fetchall()

    df = pd.DataFrame(
        data,
        columns=[
            "ID",
            "Timestamp",
            "Employee ID",
            "Emotion",
            "Recommended Task"
        ]
    )

    st.dataframe(df)

    st.subheader("Emotion Distribution")
    st.bar_chart(df["Emotion"].value_counts())


