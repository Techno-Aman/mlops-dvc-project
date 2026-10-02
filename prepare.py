from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

data = fetch_california_housing(as_frame=True)

df = data.data.copy()
df['MedHouseVal'] = data.target

df.to_csv("data/california_housing.csv",index=False)
X=df.drop(columns=["MedHouseVal"])
y=df["MedHouseVal"]


X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

train_data = X_train.copy()
train_data["MedHouseVal"] = y_train

test_data = X_test.copy()
test_data["MedHouseVal"] = y_test

train_data.to_csv("data/train.csv", index=False)
test_data.to_csv("data/test.csv", index=False)

# train = pd.read_csv("data/train.csv")
# test = pd.read_csv("data/test.csv")

# print(train.shape)
# print(test.shape)
# print(train.head())