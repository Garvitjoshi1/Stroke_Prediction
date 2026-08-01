import mlflow


class MLFlowTracker:

    def __init__(
        self,
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="NeuroGuard",
    ):

        mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

    def start_run(
        self,
        run_name=None,
    ):

        return mlflow.start_run(
            run_name=run_name
        )

    def end_run(self):

        mlflow.end_run()