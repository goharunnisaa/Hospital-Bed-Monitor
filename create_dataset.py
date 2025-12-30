import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)
np.random.seed(42)

dates = pd.date_range(start="2024-01-01", end="2024-06-30", freq="D")

hospitals = {
    "AIIMS Delhi": 200,
    "Apollo Chennai": 175,
    "Care Hospitals Hyderabad": 160,
    "KIMS Vijayawada": 130,
    "Fortis Bangalore": 185,
    "Max Healthcare Delhi": 190
}

data = []

for hospital, capacity in hospitals.items():
    base = capacity * 0.75
    for date in dates:
        weekday = date.weekday()  # Weekend more load
        variation = np.random.randint(-10, 15)
        weekend_boost = 10 if weekday >= 5 else 0
        occupied = int(min(capacity, base + variation + weekend_boost))
        data.append([hospital, date, occupied, capacity])

df = pd.DataFrame(data, columns=["Hospital", "Date", "Bed_Used", "Total_Beds"])
df.to_csv("data/hospital_data.csv", index=False)

print("📌 Realistic dataset updated!")
