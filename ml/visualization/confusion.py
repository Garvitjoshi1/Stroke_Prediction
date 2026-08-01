from __future__ import annotations

import logging

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from ml.visualization.base import BasePlotter


logger = logging.getLogger(__name__)


class ConfusionMatrixPlotter(BasePlotter):

    def __init__(self):

        super().__init__()

    def plot(
        self,
        model,
        X_test,
        y_test,
        normalize: bool = False,
    ):

        logger.info("Generating Confusion Matrix...")

        predictions = model.predict(X_test)

        cm = confusion_matrix(
            y_test,
            predictions
        )

        display_matrix = cm.astype(float)

        if normalize:
            display_matrix = (
                display_matrix /
                display_matrix.sum(axis=1, keepdims=True)
            )

        fig, ax = self.create_figure(
            figsize=(6, 6)
        )

        image = ax.imshow(
            display_matrix,
            interpolation="nearest"
        )

        plt.colorbar(
            image,
            ax=ax
        )

        class_names = [
            "No Stroke",
            "Stroke"
        ]

        ax.set_xticks(np.arange(2))
        ax.set_yticks(np.arange(2))

        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        ax.set_title("Confusion Matrix")

        threshold = display_matrix.max() / 2

        for i in range(cm.shape[0]):

            for j in range(cm.shape[1]):

                if normalize:

                    text = (
                        f"{cm[i,j]}\n"
                        f"{display_matrix[i,j]:.2%}"
                    )

                else:

                    text = str(cm[i,j])

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white"
                    if display_matrix[i, j] > threshold
                    else "black",
                    fontsize=11,
                    fontweight="bold"
                )

        fig.tight_layout()

        return fig

    def plot_and_save(
        self,
        model,
        X_test,
        y_test,
        filename="confusion_matrix.png",
        normalize=False,
    ):

        fig = self.plot(
            model,
            X_test,
            y_test,
            normalize=normalize,
        )

        return self.finish(
            fig,
            filename,
        )