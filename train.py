import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import glob

X = []
y = []

files = glob.glob("*.pkl")

for file in files:
    
    if file == "model.pkl":
        continue

    label = file.replace(".pkl", "")
    data = pickle.load(open(file, "rb"))

    for row in data:
        X.append(row)
        y.append(label)

X = np.array(X)
y = np.array(y)

model = RandomForestClassifier()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained successfully!")