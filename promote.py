from mlflow import MlflowClient

client = MlflowClient()

champion = client.get_model_version_by_alias(
    name = "CaliforniaHousingRegressor",
    alias= "champion"
)

print("champion version :",champion.version)
print("champion run id :",champion.run_id)

champion_run = client.get_run(champion.run_id)
champion_rmse = champion_run.data.metrics["rmse"]
print("champion rmse" , champion_rmse)

candidate = client.get_latest_versions(
    name="CaliforniaHousingRegressor",
    stages=["None"]
)[-1]

print("Candidate version:", candidate.version)
print("Candidate run ID:", candidate.run_id)

candidate_run = client.get_run(candidate.run_id)

candidate_rmse = candidate_run.data.metrics["rmse"]

print("Champion RMSE:", champion_rmse)
print("Candidate RMSE:", candidate_rmse)

minimum_improvement = 0.001

if candidate_rmse < champion_rmse - minimum_improvement:
    client.set_registered_model_alias(
        name="CaliforniaHousingRegressor",
        alias="previous_champion",
        version=champion.version
    )

    client.set_registered_model_alias(
        name="CaliforniaHousingRegressor",
        alias="champion",
        version=candidate.version
    )

    print(f"Promoted version {candidate.version} to champion.")
else:
    print("Candidate did not improve enough. Keeping current champion.")