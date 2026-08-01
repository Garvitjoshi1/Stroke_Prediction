from __future__ import annotations

import logging
from pathlib import Path

import joblib
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

logger = logging.getLogger(__name__)


class SHAPExplainer:

    def __init__(
        self,
        model=None,
        output_dir="artifacts/explainability",
    ):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if model is None:

            logger.info(
                "Loading trained model..."
            )

            artifact = joblib.load(
                "artifacts/models/logistic_regression.joblib"
            )

            if isinstance(artifact, dict):

                self.model = artifact["model"]

            else:

                self.model = artifact

        else:

            self.model = model

        self.explainer = None

        self.background_data = None

    def _build_linear(
        self,
        X,
    ):

        logger.info(
            "Using SHAP LinearExplainer."
        )

        return shap.LinearExplainer(
            self.model,
            X,
        )

    def _build_tree(self):

        logger.info(
            "Using SHAP TreeExplainer."
        )

        return shap.TreeExplainer(
            self.model
        )

    def _build_generic(
        self,
        X,
    ):

        logger.info(
            "Using generic SHAP Explainer."
        )

        return shap.Explainer(
            self.model.predict,
            X,
        )

    def build(
        self,
        X,
    ):

        if isinstance(
            self.model,
            (
                LogisticRegression,
                LinearRegression,
            ),
        ):

            self.explainer = self._build_linear(X)

        elif isinstance(
            self.model,
            (
                RandomForestClassifier,
                RandomForestRegressor,
                ExtraTreesClassifier,
                ExtraTreesRegressor,
                GradientBoostingClassifier,
                GradientBoostingRegressor,
            ),
        ):

            self.explainer = self._build_tree()

        else:

            self.explainer = self._build_generic(X)

        logger.info(
            "SHAP explainer ready."
        )

        return self.explainer

    def explain(
        self,
        X,
    ):

        if self.explainer is None:

            self.build(X)

        logger.info(
            "Calculating SHAP values..."
        )

        shap_values = self.explainer(X)

        logger.info(
            "SHAP values generated."
        )

        return shap_values