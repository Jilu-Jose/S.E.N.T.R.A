<div align="center">

<br/>

```
███████╗███████╗███╗   ██╗████████╗██████╗  █████╗
██╔════╝██╔════╝████╗  ██║╚══██╔══╝██╔══██╗██╔══██╗
███████╗█████╗  ██╔██╗ ██║   ██║   ██████╔╝███████║
╚════██║██╔══╝  ██║╚██╗██║   ██║   ██╔══██╗██╔══██║
███████║███████╗██║ ╚████║   ██║   ██║  ██║██║  ██║
╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝
```

# Smart Emotion & Task Recommendation Analytics

**Real-time facial emotion detection paired with intelligent task recommendations to improve workforce well-being**

<br/>

[![Live Demo](https://img.shields.io/badge/Live_Demo-s--e--n--t--r--a.onrender.com-0A66C2?style=for-the-badge&logo=googlechrome&logoColor=white)](https://s-e-n-t-r-a.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)

<br/>

</div>

---

## ![overview](https://img.shields.io/badge/-Overview-1a1a2e?style=flat-square&logo=readme&logoColor=white) Overview

**S.E.N.T.R.A** is a real-time Emotion Detection and Task Recommendation System that analyzes facial expressions using a deep learning model (Mini-Xception) to evaluate current stress levels and logs detected emotions with timestamps. Integrated with an HR alert engine, S.E.N.T.R.A provides smart task recommendations based on real-time mood analysis to improve workforce well-being.

---

## ![features](https://img.shields.io/badge/-Key_Features-1a1a2e?style=flat-square&logo=todoist&logoColor=white) Key Features

| | Feature | Description |
|---|---|---|
| ![](https://img.shields.io/badge/Live_Emotion_Detection-5C3EE8?style=flat-square&logo=opencv&logoColor=white) | **Live Emotion Detection** | Evaluates facial expressions in real-time using a lightweight Mini-Xception deep learning model |
| ![](https://img.shields.io/badge/Smart_Task_Recommendation-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) | **Smart Task Recommendation** | Random Forest classifier suggests tasks based on emotional state, workload, and deadline pressure |
| ![](https://img.shields.io/badge/Stress_Alerts-EA4335?style=flat-square&logo=gmail&logoColor=white) | **Stress Alerts** | Monitors prolonged stress indicators and triggers real-time HR email notifications |
| ![](https://img.shields.io/badge/Analytics_Dashboard-4285F4?style=flat-square&logo=chartdotjs&logoColor=white) | **Analytics Dashboard** | Visualizes emotion distributions, 24-hour trends, and hourly activity |
| ![](https://img.shields.io/badge/Automated_Logging-003B57?style=flat-square&logo=sqlite&logoColor=white) | **Automated Logging** | Tracks mood history securely in a local SQLite database for future analysis |

---

## ![ml](https://img.shields.io/badge/-How_It_Works-1a1a2e?style=flat-square&logo=scikitlearn&logoColor=white) How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                    Live Camera Feed                          │
└──────────────────────┬───────────────────────────────────────┘
                       │
         ┌─────────────▼──────────────┐
         │   OpenCV Face Detection    │
         └─────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────┐
         │  Mini-Xception CNN Model   │
         │  (FER-2013 trained)        │
         └──────┬──────────┬──────────┘
                │          │
    ┌───────────▼──┐   ┌───▼────────────────┐
    │ Emotion Tag  │   │  Stress Classifier  │
    │ + Timestamp  │   │  (Sad/Anger/Fear/   │
    │  → SQLite    │   │   Disgust trigger)  │
    └───────────┬──┘   └───┬────────────────┘
                │          │
    ┌───────────▼──┐   ┌───▼────────────────┐
    │   Analytics  │   │  HR Email Alert    │
    │   Dashboard  │   │  (SMTP / Gmail)    │
    └──────────────┘   └────────────────────┘
                       │
         ┌─────────────▼──────────────┐
         │  Random Forest Recommender │
         │  Task suggestion based on  │
         │  mood + workload + deadline│
         └────────────────────────────┘
```

---

## ![stack](https://img.shields.io/badge/-Tech_Stack-1a1a2e?style=flat-square&logo=stackshare&logoColor=white) Tech Stack

```
┌──────────────┐  ┌──────────────────────────────────────────────┐
│  FRONTEND    │  │  HTML5 · CSS3 · Vanilla JS · Chart.js        │
├──────────────┤  ├──────────────────────────────────────────────┤
│  BACKEND     │  │  Python · Flask · SQLite3                    │
├──────────────┤  ├──────────────────────────────────────────────┤
│  ML / CV     │  │  TensorFlow · Keras · Scikit-Learn · OpenCV  │
├──────────────┤  ├──────────────────────────────────────────────┤
│  DATA        │  │  NumPy · Pandas · FER-2013 Dataset           │
├──────────────┤  ├──────────────────────────────────────────────┤
│  DEPLOYMENT  │  │  Gunicorn · Render                           │
└──────────────┘  └──────────────────────────────────────────────┘
```

![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=flat-square&logo=chartdotjs&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Gunicorn](https://img.shields.io/badge/Gunicorn-499848?style=flat-square&logo=gunicorn&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=white)

---

## ![start](https://img.shields.io/badge/-Quick_Start-1a1a2e?style=flat-square&logo=dependabot&logoColor=white) Quick Start (Local Setup)

### Prerequisites

![Python](https://img.shields.io/badge/Python-3.x_required-3776AB?style=flat-square&logo=python&logoColor=white)
![pip](https://img.shields.io/badge/pip-package_manager-3775A9?style=flat-square&logo=pypi&logoColor=white)
![Gmail](https://img.shields.io/badge/Gmail-App_Password_required-EA4335?style=flat-square&logo=gmail&logoColor=white)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Jilu-Jose/EMO-DETECTION-AMDOX.git
cd EMO-DETECTION-AMDOX
```

**2. Set up a virtual environment** *(recommended to avoid dependency conflicts)*
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file in the root directory:
```env
EMAIL_ID=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
HR_EMAIL=hr_email@example.com
```

**5. Run the application**
```bash
python app.py
```

Navigate to **`http://localhost:10000`** in your browser.

---

## ![gmail](https://img.shields.io/badge/-Gmail_App_Password_Setup-1a1a2e?style=flat-square&logo=gmail&logoColor=white) Gmail App Password Setup

S.E.N.T.R.A requires a Gmail **App Password** to send automated HR alerts. Standard account passwords will not work.

| Step | Action |
|---|---|
| **1** | Go to [Google Account Settings](https://myaccount.google.com/) |
| **2** | Navigate to the **Security** tab |
| **3** | Ensure **2-Step Verification** is enabled |
| **4** | Search for **App passwords** in the search bar |
| **5** | Enter a custom name (e.g., `SENTRA App`) and click **Create** |
| **6** | Copy the generated **16-character password** into `.env` under `EMAIL_PASSWORD` |

---

## ![deploy](https://img.shields.io/badge/-Deployment-1a1a2e?style=flat-square&logo=render&logoColor=white) Deployment

S.E.N.T.R.A is configured for deployment on **Render** or **Heroku**.

### Deploying to Render

**1. Push your code to GitHub**

**2. Create a new Web Service on [Render](https://render.com/)**

**3. Connect your GitHub repository**

**4. Set the build command:**
```bash
pip install -r requirements.txt
```

**5. Set the start command:**
```bash
gunicorn app:app
```

**6. Add environment variables** in the Render Dashboard under **Environment**:

| Key | Value |
|---|---|
| `EMAIL_ID` | `your_email@gmail.com` |
| `EMAIL_PASSWORD` | `your_16_char_app_password` |
| `HR_EMAIL` | `hr_email@example.com` |

**7. Click Deploy**

---

## ![structure](https://img.shields.io/badge/-Project_Structure-1a1a2e?style=flat-square&logo=files&logoColor=white) Project Structure

```
EMO-DETECTION-AMDOX/
│
├── app.py                     # Main Flask application entry point
├── database.py                # Database connection & table creation
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── .env                       # Environment variables (not tracked)
│
├── models/
│   ├── fer2013_mini_XCEPTION.102-0.66.hdf5   # Facial emotion recognition model
│   └── task_recommender_rf.pkl               # Task recommendation model
│
└── templates/
    ├── index.html             # Main application view (live feed + dashboard)
    ├── landing.html           # Introduction / landing page
    └── login.html             # User login interface
```

---

## ![ack](https://img.shields.io/badge/-Acknowledgements-1a1a2e?style=flat-square&logo=opensourceinitiative&logoColor=white) Acknowledgements

| Resource | Details |
|---|---|
| ![](https://img.shields.io/badge/Mini--Xception_Architecture-grey?style=flat-square&logo=github&logoColor=white) | Facial Emotion Recognition model by [@oarriaga](https://github.com/oarriaga/face_classification) |
| ![](https://img.shields.io/badge/FER--2013_Dataset-grey?style=flat-square&logo=kaggle&logoColor=white) | Training dataset from [Kaggle FER-2013 Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) |

---

<div align="center">

Built to improve workforce well-being through intelligent emotion-aware systems

[![Status](https://img.shields.io/badge/status-live-brightgreen?style=flat-square&logo=googlechrome&logoColor=white)](https://s-e-n-t-r-a.onrender.com)
[![Made with Python](https://img.shields.io/badge/Made_with-Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Powered by TensorFlow](https://img.shields.io/badge/Powered_by-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)

</div>