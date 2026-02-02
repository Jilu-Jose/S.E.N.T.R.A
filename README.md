# EMO-DETECTION-AMDOX

A real-time Emotion Detection System that analyzes facial expressions using a deep learning model and logs detected emotions with timestamps in a local database. The project is built using Python and focuses on clean structure, reproducibility, and practical deployment.

---

## Email Configuration (Important)

This project sends email alerts, so you must provide your own email credentials in the `app.py` file before running the application.

### Step 1: Open `app.py`
Locate and open the `app.py` file in the project root directory.

### Step 2: Update Email Credentials
Find the email configuration section and replace the placeholders with your details:

```python
EMAIL_ID = "your_email@example.com"
EMAIL_PASSWORD = "your_email_app_password"
HR_EMAIL = "receiver_email@example.com"


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
  - HTML5/CSS3
  - JavaScript  

---

## Project Structure

Main-Emo/
│
|__ templates
|    |__ index.html     
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


