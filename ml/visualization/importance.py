from __future__ import annotations

import logging

import numpy as np

from sklearn.inspection import permutation_importance

from ml.visualization.base import BasePlotter


logger = logging.getLogger(__name__)


class FeatureImportancePlotter(BasePlotter):
    
    def __init__(self):

        super().__init__()

    def get_importance(
        self,
        model,
        X_test,
        y_test,
        feature_names,
    ):

        logger.info("Calculating feature importance...")

        if hasattr(model, "feature_importances_"):

            importance = model.feature_importances_

        elif hasattr(model, "coef_"):

            importance = np.abs(model.coef_[0])

        else:

            logger.info(
                "Using permutation importance..."
            )

            result = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=10,
                random_state=42,
                scoring="roc_auc",
            )

            importance = result.importances_mean

        return (
            np.array(feature_names),
            np.array(importance),
        )

    def plot(
        self,
        model,
        X_test,
        y_test,
        feature_names,
        top_k=15,
    ):

        features, importance = self.get_importance(
            model,
            X_test,
            y_test,
            feature_names,
        )

        order = np.argsort(
            importance
        )[::-1]

        order = order[:top_k]

        features = features[order]

        importance = importance[order]

        fig, ax = self.create_figure(
            figsize=(10, 7)
        )

        ax.barh(
            features[::-1],
            importance[::-1],
        )

        self.style_axis(
            ax,
            title=f"Top {top_k} Feature Importance",
            xlabel="Importance",
            ylabel="Feature",
            grid=False,
        )

        return fig

    def plot_and_save(
        self,
        model,
        X_test,
        y_test,
        feature_names,
        filename="feature_importance.png",
    ):

        fig = self.plot(
            model,
            X_test,
            y_test,
            feature_names,
        )

        return self.finish(
            fig,
            filename,
        )