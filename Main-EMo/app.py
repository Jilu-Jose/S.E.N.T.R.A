import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from database import conn, cursor

st.set_page_config(page_title="Employee Emotion Monitor", layout="wide")


model = load_model(
    "models/fer2013_mini_XCEPTION.102-0.66.hdf5",
    compile=False
)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


task_map = {
    "Happy": "Creative brainstorming or teamwork",
    "Neutral": "Routine or documentation work",
    "Sad": "Light tasks or peer support",
    "Angry": "Break or physical activity",
    "Fear": "Guided low-pressure tasks",
    "Surprise": "Learning or exploration",
    "Disgust": "Task rotation or rest"
}

stress_emotions = ["Sad", "Angry", "Fear", "Disgust"]

if "stress_count" not in st.session_state:
    st.session_state.stress_count = 0

menu = st.sidebar.selectbox(
    "Dashboard Menu",
    ["Emotion Detection", "Mood History"]
)

st.title("S.E.N.T.R.A System")


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

                preds = model.predict(face, verbose=0)
                emotion = emotion_labels[np.argmax(preds)]

                
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, emotion, (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                
                recommended_task = task_map[emotion]
                emotion_placeholder.markdown(f"### Detected Emotion: **{emotion}**")

                task_placeholder.success(f"Recommended Task: {recommended_task}")


                
                cursor.execute(
                    "INSERT INTO mood_logs VALUES (NULL, ?, ?, ?)",
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     "EMP001",
                     emotion)
                )
                conn.commit()

                
                if emotion in stress_emotions:
                    st.session_state.stress_count += 1
                else:
                    st.session_state.stress_count = 0

                if st.session_state.stress_count >= 5:
                    alert_placeholder.error("Stress Alert: HR Notification Triggered")
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
        columns=["ID", "Timestamp", "Employee ID", "Emotion"]
    )

    st.dataframe(df)

    st.subheader("Emotion Distribution")
    st.bar_chart(df["Emotion"].value_counts())





























    