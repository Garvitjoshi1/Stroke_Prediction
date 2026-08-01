from pathlib import Path


class Settings:

    APP_NAME = "NeuroGuard Stroke Prediction API"

    VERSION = "1.0.0"

    DESCRIPTION = (
        "Production API for Stroke Risk Prediction"
    )

    ARTIFACT_DIR = Path("artifacts")

    MODEL_NAME = "logistic_regression.joblib"

    HOST = "0.0.0.0"

    PORT = 8000

    DEBUG = True


settings = Settings()