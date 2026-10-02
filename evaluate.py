from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error
import joblib
import pandas as pd
import json
import mlflow

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("CaliforniaHousing")

with open("run_id.txt", "r") as file:
    run_id = file.read().strip()

with mlflow.start_run(run_id=run_id) :
    data = pd.read_csv("data/test.csv")

    X_test = data.drop(columns=["MedHouseVal"])
    y_test = data["MedHouseVal"]

    model = joblib.load("model.pkl")

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test,y_pred) 
    rmse = root_mean_squared_error(y_test,y_pred) 
    r2 = r2_score(y_test,y_pred) 

    metrics = {
        "mae" : mae,
        "rmse" : rmse,
        "r2" : r2,
    }

    mlflow.log_metrics(metrics)
    print("Logged metrics:", metrics)
    print("Run ID:", mlflow.active_run().info.run_id)

with open("metrics.json","w") as file :
    json.dump(metrics, file , indent=4)