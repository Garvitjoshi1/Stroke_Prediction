import mlflow


class MLFlowRegistry:

    def register(
        self,
        model_uri,
        name,
    ):

        mlflow.register_model(
            model_uri=model_uri,
            name=name,
        )