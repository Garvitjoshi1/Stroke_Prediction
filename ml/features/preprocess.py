from pathlib import Path
import logging
import joblib

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

class DataPreprocessor:

    def __init__(self):

        self.numeric_features = None

        self.categorical_features = None

        self.numeric_pipeline = None

        self.categorical_pipeline = None

        self.preprocessor = None

    def detect_features(self, X: pd.DataFrame):

        logger.info("Detecting feature types...")

        self.numeric_features = (
            X.select_dtypes(
                include=["number"]
            )
            .columns
            .tolist()
        )

        self.categorical_features = (
            X.select_dtypes(
                include=["object", "string"]
            )
            .columns
            .tolist()
        )

        logger.info(
            f"Numerical Features ({len(self.numeric_features)}): "
            f"{self.numeric_features}"
        )

        logger.info(
            f"Categorical Features ({len(self.categorical_features)}): "
            f"{self.categorical_features}"
        )

        return (
            self.numeric_features,
            self.categorical_features
        )

    def build_numeric_pipeline(self):

        logger.info(
            "Building numerical preprocessing pipeline..."
        )

        self.numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),

                (
                    "scaler",
                    StandardScaler()
                ),
            ]
        )

        return self.numeric_pipeline

    def build_categorical_pipeline(self):

        logger.info(
            "Building categorical preprocessing pipeline..."
        )

        self.categorical_pipeline = Pipeline(
            steps=[

                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),

                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )

            ]
        )

        return self.categorical_pipeline

    def build_preprocessor(self):

        logger.info(
            "Building ColumnTransformer..."
        )

        if self.numeric_pipeline is None:
            raise ValueError(
                "Numeric pipeline has not been created."
            )

        if self.categorical_pipeline is None:
            raise ValueError(
                "Categorical pipeline has not been created."
            )

        self.preprocessor = ColumnTransformer(

            transformers=[

                (
                    "numerical",
                    self.numeric_pipeline,
                    self.numeric_features
                ),

                (
                    "categorical",
                    self.categorical_pipeline,
                    self.categorical_features
                )

            ],

            remainder="drop"

        )

        logger.info(
            "ColumnTransformer created successfully."
        )

    def fit(self, X: pd.DataFrame):

        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor has not been built. "
                "Call build_preprocessor() first."
            )

        logger.info("Fitting preprocessor on training data...")

        self.preprocessor.fit(X)

        logger.info("Preprocessor fitted successfully.")

        return self

    def transform(self, X: pd.DataFrame):

        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor has not been fitted."
            )

        logger.info("Transforming dataset...")

        X_processed = self.preprocessor.transform(X)
        feature_names = self.get_feature_names()

        X_processed = pd.DataFrame(
            X_processed,
            columns=feature_names,
            index=X.index
        )

        logger.info(
            f"Transformation completed. Shape: {X_processed.shape}"
        )

        return X_processed

    def fit_transform(self, X: pd.DataFrame):

        logger.info("Running fit_transform...")

        self.fit(X)

        return self.transform(X)

    def get_feature_names(self):

        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor has not been fitted."
            )

        feature_names = self.preprocessor.get_feature_names_out()

        logger.info(
            f"Generated {len(feature_names)} feature names."
        )

        return feature_names

    def save_preprocessor(
        self,
        output_dir="artifacts/preprocessors"
    ):

        if self.preprocessor is None:
            raise ValueError(
                "Nothing to save. Fit the preprocessor first."
            )

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        save_path = output_dir / "preprocessor.joblib"

        joblib.dump(
    {
        "preprocessor": self.preprocessor,
        "feature_names": self.get_feature_names().tolist(),
        "numeric_features": self.numeric_features,
        "categorical_features": self.categorical_features,
    },
    save_path,
    )
        
        logger.info(
            f"Preprocessor saved to {save_path.resolve()}"
        )

        return save_path

    def load_preprocessor(
        self,
        path="artifacts/preprocessors/preprocessor.joblib"
    ):

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        self.preprocessor = joblib.load(path)

        logger.info(
            f"Loaded preprocessor from {path.resolve()}"
        )

        return self.preprocessor