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
        artifact = joblib.load(path)
        if isinstance(artifact, dict):

            logger.info(
                "Detected bundled model artifact."
            )

            return artifact
        logger.info(
            "Detected legacy model artifact."
        )

        return {

            "model": artifact,

            "model_name": "logistic_regression",

            "training_time": None,

        }

    def load_preprocessor(
        self,
        filename: str = "preprocessor.joblib",
    ):

        path = self.preprocessor_dir / filename

        logger.info(
            "Loading preprocessor from %s",
            path,
        )

        artifact = joblib.load(path)

        if isinstance(artifact, dict):

            logger.info(
                "Detected bundled preprocessor artifact."
            )

            return artifact

        logger.info(
            "Detected legacy preprocessor artifact."
        )

        return {

            "preprocessor": artifact,

            "feature_names": None,

            "numeric_features": None,

            "categorical_features": None,

        }

    def load_threshold(
        self,
        filename: str = "threshold.json",
    ):

        path = self.metadata_dir / filename

        if not path.exists():

            logger.warning(
                "Threshold file missing. Using default 0.50"
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

            logger.warning(
                "Model info file missing."
            )

            return {}

        with open(path, "r") as f:

            return json.load(f)

    def load_all(self):

        logger.info("=" * 60)
        logger.info("Loading inference artifacts...")
        logger.info("=" * 60)

        model_bundle = self.load_model()

        preprocessor_bundle = self.load_preprocessor()

        artifacts = {

            "model":
                model_bundle["model"],

            "model_name":
                model_bundle.get("model_name"),

            "training_time":
                model_bundle.get("training_time"),

            "preprocessor":
                preprocessor_bundle["preprocessor"],

            "feature_names":
                preprocessor_bundle["feature_names"],

            "numeric_features":
                preprocessor_bundle["numeric_features"],

            "categorical_features":
                preprocessor_bundle["categorical_features"],

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

        print(f"{key:20}: {type(value)}")