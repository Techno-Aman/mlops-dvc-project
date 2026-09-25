import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load the trained model
model = joblib.load("model.pkl")

# Load test data
data = pd.read_csv("data/test.csv")

# Separate features and actual labels
X_test = data.drop(columns=["species"])
y_test = data["species"]

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Save the metric
with open("metrics.txt", "w") as file:
    file.write(f"accuracy: {accuracy}\n")

print("Evaluation complete!")
print("Accuracy:", accuracy)