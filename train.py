import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import yaml

# Load training data
data = pd.read_csv("data/train.csv")
param = yaml.safe_load(open("params.yaml", "r")) 

# Separate features and target
X = data.drop(columns=["species"])
y = data["species"]

# Train the model
model = RandomForestClassifier(
    n_estimators=param["train"]["n_estimators"],
    random_state=42
)

model.fit(X, y)

# Save the trained model
joblib.dump(model, "model.pkl")

print("Model training complete!")