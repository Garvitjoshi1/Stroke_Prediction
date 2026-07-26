from pathlib import Path
import logging
import time
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from ml.models.registry import get_model_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class ModelTrainer:

    def __init__(self):

        self.models = get_model_registry()

        self.model = None
        self.model_name = None
        self.training_time = None

    def train(
        self,
        model_name: str,
        X_train,
        y_train
    ):

        if model_name not in self.models:
            raise ValueError(
                f"Unknown model: {model_name}"
            )

        self.model_name = model_name
        self.model = self.models[model_name]

        logger.info(
            f"Training {model_name}..."
        )

        start = time.perf_counter()

        self.model.fit(
            X_train,
            y_train
        )

        end = time.perf_counter()

        self.training_time = end - start

        logger.info(
            f"{model_name} training completed."
        )

        logger.info(
            f"Training Time: {self.training_time:.3f} sec"
        )

        return self.model
    
    def predict(
        self,
        X
    ):

        if self.model is None:
            raise ValueError(
                "Train or load a model first."
            )

        return self.model.predict(X)

    def predict_proba(
        self,
        X
    ):

        if self.model is None:
            raise ValueError(
                "Train or load a model first."
            )

        if not hasattr(
            self.model,
            "predict_proba"
        ):
            raise AttributeError(
                "Model doesn't support predict_proba."
            )

        return self.model.predict_proba(X)

    def save_model(
        self,
        output_dir="artifacts/models"
    ):

        if self.model is None:
            raise ValueError(
                "No trained model found."
            )

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        save_path = output_dir / f"{self.model_name}.joblib"

        joblib.dump(

            {

                "model": self.model,

                "model_name": self.model_name,

                "training_time": self.training_time

            },

            save_path

        )

        logger.info(
            f"Model saved to {save_path.resolve()}"
        )

        return save_path

    def load_model(
        self,
        path
    ):

        path = Path(path)

        checkpoint = joblib.load(path)

        self.model = checkpoint["model"]

        self.model_name = checkpoint["model_name"]

        self.training_time = checkpoint["training_time"]

        logger.info(
            f"Loaded model: {self.model_name}"
        )

        return self.model