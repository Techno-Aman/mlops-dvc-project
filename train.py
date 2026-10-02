from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd
import yaml
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment("CaliforniaHousing")

with open("params.yaml", "r") as file :
        params = yaml.safe_load(file)

with mlflow.start_run() :
    with open("run_id.txt", "w") as file :
          file.write(mlflow.active_run().info.run_id)
    run_id = mlflow.active_run().info.run_id

    mlflow.log_params({
        "n_estimators": params["train"]["n_estimators"],
        "random_state": params["train"]["random_state"]
    })
   
    data = pd.read_csv("data/train.csv")

    X = data.drop(columns=["MedHouseVal"])
    y = data["MedHouseVal"]

    model = RandomForestRegressor(
            n_estimators=params["train"]["n_estimators"],
            random_state=params["train"]["random_state"]
        )

    model.fit(X, y)

    mlflow.sklearn.log_model(
        model,
        "model",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

model_uri = f"runs:/{run_id}/model"

model_version = mlflow.register_model(
      model_uri=model_uri,
      name="CaliforniaHousingRegressor"
)
# client = MlflowClient()

# client.set_registered_model_alias(
#     name="CaliforniaHousingRegressor",
#     alias="champion",
#     version=model_version.version
# )

joblib.dump(model,"model.pkl")