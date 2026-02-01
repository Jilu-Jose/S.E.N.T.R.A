# EMO-DETECTION-AMDOX

A real-time Emotion Detection System that analyzes facial expressions using a deep learning model and logs detected emotions with timestamps in a local database. The project is built using Python and focuses on clean structure, reproducibility, and practical deployment.

---

## Features

- Real-time emotion detection using webcam input  
- Deep learning based facial emotion recognition  
- Automatic emotion logging with timestamps  
- Lightweight SQLite database  
- Simple script to view stored emotion records  
- Clean and modular project structure  

---

## Tech Stack

- Language: Python  
- Libraries and Frameworks:
  - OpenCV  
  - TensorFlow / Keras  
  - NumPy  
  - SQLite3  
  - Streamlit  

---

## Project Structure

Main-Emo/
│
├── app.py # Main application entry point
├── emotion_dec.py # Emotion detection logic
├── database.py # Database connection and table creation
├── view_db.py # Script to view stored emotion records
├── models/ # Pre-trained emotion detection model
├── requirements.txt # Project dependencies
├── .gitignore # Ignored files and folders
└── README.md # Project documentation



---

## Ignored Files

The following files and folders are intentionally excluded from version control:

- venv/ (virtual environment)  
- *.db (database files)  
- __pycache__/ (Python cache files)  
- .env (environment variables)

The database file is created automatically when the application runs.

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/EMO-DETECTION-AMDOX.git
cd EMO-DETECTION-AMDOX


