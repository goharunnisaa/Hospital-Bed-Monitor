import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv("data/hospital_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Day"] = df["Date"].dt.dayofyear
df["Weekday"] = df["Date"].dt.weekday

X = df[["Day", "Weekday"]]
y = df["Bed_Used"]

model = RandomForestRegressor(n_estimators=320, random_state=42)
model.fit(X, y)

with open("models.pkl", "wb") as f:
    pickle.dump(model, f)

print("🤖 Model retrained with improved prediction!")
