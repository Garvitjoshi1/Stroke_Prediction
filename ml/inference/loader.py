from pathlib import Path
import json
import logging

import joblib

logger = logging.getLogger(__name__)


class ArtifactLoader:

    def __init__(
        self,
        artifact_dir: str = "artifacts",
    ):

        self.artifact_dir = Path(artifact_dir)

        self.model_dir = self.artifact_dir / "models"
        self.preprocessor_dir = self.artifact_dir / "preprocessors"
        self.metadata_dir = self.artifact_dir / "metadata"

    def load_model(
        self,
        filename: str = "logistic_regression.joblib",
    ):

        path = self.model_dir / filename

        logger.info(
            "Loading model from %s",
            path,
        )

        model = joblib.load(path)

        logger.info("Model loaded successfully.")

        return model

    def load_preprocessor(
        self,
        filename: str = "preprocessor.joblib",
    ):

        path = self.preprocessor_dir / filename

        logger.info(
            "Loading preprocessor from %s",
            path,
        )

        preprocessor = joblib.load(path)

        logger.info("Preprocessor loaded successfully.")

        return preprocessor

    def load_feature_names(
        self,
        filename: str = "feature_names.joblib",
    ):

        path = self.preprocessor_dir / filename

        if not path.exists():

            logger.warning(
                "Feature names not found."
            )

            return None

        logger.info(
            "Loading feature names..."
        )

        return joblib.load(path)

    def load_threshold(
        self,
        filename: str = "threshold.json",
    ):

        path = self.metadata_dir / filename

        if not path.exists():

            logger.warning(
                "Threshold file missing. Using 0.50"
            )

            return 0.50

        with open(path, "r") as f:

            data = json.load(f)

        threshold = data.get(
            "threshold",
            0.50,
        )

        logger.info(
            "Threshold loaded: %.3f",
            threshold,
        )

        return threshold

    def load_model_info(
        self,
        filename: str = "model_info.json",
    ):

        path = self.metadata_dir / filename

        if not path.exists():

            return {}

        with open(path, "r") as f:

            return json.load(f)

    def load_all(self):

        logger.info(
            "=" * 60
        )
        logger.info(
            "Loading inference artifacts..."
        )

        artifacts = {

            "model":
                self.load_model(),

            "preprocessor":
                self.load_preprocessor(),

            "feature_names":
                self.load_feature_names(),

            "threshold":
                self.load_threshold(),

            "model_info":
                self.load_model_info(),

        }

        logger.info(
            "Inference artifacts loaded successfully."
        )

        return artifacts


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    loader = ArtifactLoader()

    artifacts = loader.load_all()

    print()

    print("=" * 60)
    print("Loaded Artifacts")
    print("=" * 60)

    for key, value in artifacts.items():

        print(f"{key:15}: {type(value)}")