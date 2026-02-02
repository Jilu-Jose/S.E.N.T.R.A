import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib


df = pd.read_csv("task_data1.csv")

print(df.head())


X = df.drop("Recommended_Task", axis=1)
y = df["Recommended_Task"]


mood_encoder = LabelEncoder()
task_encoder = LabelEncoder()

X["Mood"] = mood_encoder.fit_transform(X["Mood"])
y = task_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


ml = RandomForestClassifier(n_estimators=200,random_state=42)

ml.fit(X_train, y_train)


joblib.dump(
    {
        "model": ml,
        "mood_encoder": mood_encoder,
        "task_encoder": task_encoder
    },
    "task_recommender_rf.pkl"
)


