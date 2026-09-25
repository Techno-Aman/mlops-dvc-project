import pandas as pd
from sklearn.model_selection import train_test_split

# Load our dataset
data = pd.read_csv("data/iris.csv")

# Separate features and target
X = data.drop(columns=["species"])
y = data["species"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Save the split datasets
train_data = X_train.copy()
train_data["species"] = y_train

test_data = X_test.copy()
test_data["species"] = y_test

train_data.to_csv("data/train.csv", index=False)
test_data.to_csv("data/test.csv", index=False)

print("Data preparation complete!")
print("Training rows:", len(train_data))
print("Testing rows:", len(test_data))