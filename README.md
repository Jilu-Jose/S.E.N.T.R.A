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
```

## Generating Email App Password (Gmail – App Security)

To send email alerts, you must generate an **App Password** from your email provider. Below are the steps for **Gmail**.

### Step 1: Enable 2-Step Verification
1. Go to your Google Account.
2. Open **Security** settings.
3. Enable **2-Step Verification**.
   - This is mandatory to generate an app password.

### Step 2: Generate App Password
1. In Google Account → **Security**
2. Go to **App passwords**
3. Select:
   - App: **Mail**
   - Device: **Windows Computer**
4. Click **Generate**
5. Google will provide a **16-character password**

### Step 3: Use App Password in `app.py`
Paste the generated password in your code:
```python
SENDER_PASSWORD = "your_16_character_app_password"
```

## Running the Project in VS Code

### Step 1: Open Integrated Terminal
- Open Visual Studio Code
- Press Ctrl + ` (backtick)
  OR
- Go to Terminal → New Terminal

### Step 2: Change Directory
Navigate to the project folder:
```python
cd Main-EMo
```
### Step 3: Activate Virtual Environment
Activate the Python virtual environment:
```python
.\venv\Scripts\activate
```
You should see (venv) in the terminal.

### Step 4: Run the Application
Run the project using:
```python
python app.py
```

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


