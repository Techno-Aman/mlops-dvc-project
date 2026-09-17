import mlflow 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X,y = load_iris(return_X_y=True)


X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("model registry learning")

with mlflow.start_run() as run :
    model.fit(X_train,y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.sklearn.log_model(
        model,
        "model"
    )

    print("run id :",run.info.run_id)
    print("accuracy :",accuracy)


model_uri = f"runs:/{run.info.run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="IrisClassifier"
)

print("model :",registered_model.name)
print("Version :",registered_model.version)

