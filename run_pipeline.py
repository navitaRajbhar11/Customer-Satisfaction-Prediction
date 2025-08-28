from pipelines.training_pipeline import train_pipeline
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from zenml.client import Client


if __name__ == "__main__":
    # Build pipeline with keyword arguments
    training = train_pipeline(
        ingest_data=ingest_data(),
        clean_data=clean_data(),
        train_model=train_model(),
        evaluation=evaluation(),
    )

    # Run pipeline
    training.run()

    # Get MLflow tracking URI from the active experiment tracker
    client = Client()
    mlflow_tracker = client.active_stack.experiment_tracker
    tracking_uri = mlflow_tracker.get_tracking_uri()

    print(
        "✅ Pipeline run completed!\n\n"
        "Now start MLflow UI with:\n\n"
        f"    mlflow ui --backend-store-uri '{tracking_uri}'\n\n"
        "👉 Open the above link in your browser to explore experiment runs.\n"
        "Look for the `mlflow_example_pipeline` experiment to compare results."
    )
