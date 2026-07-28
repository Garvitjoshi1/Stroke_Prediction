from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any
from ml.experiment.artifact_manager import ArtifactManager

import joblib
import pandas as pd


logger = logging.getLogger(__name__)


class ExperimentManager:

    def __init__(
        self,
        root_dir: str = "artifacts/experiments",
        experiment_name: str | None = None
    ):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if experiment_name:

            folder_name = f"{experiment_name}_{timestamp}"

        else:

            folder_name = f"experiment_{timestamp}"

        self.experiment_dir = Path(root_dir) / folder_name

        self.plots_dir = self.experiment_dir / "plots"

        self.models_dir = self.experiment_dir / "models"

        self.data_dir = self.experiment_dir / "data"

        self._create_directories()
        self.artifact_manager = ArtifactManager(
            self.experiment_dir
            )

    def _create_directories(self):

        self.experiment_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.plots_dir.mkdir(
            exist_ok=True
        )

        self.models_dir.mkdir(
            exist_ok=True
        )

        self.data_dir.mkdir(
            exist_ok=True
        )

        logger.info(
            f"Experiment Directory : {self.experiment_dir}"
        )

    def save_dataframe(
        self,
        df: pd.DataFrame,
        filename: str
    ):

        path = self.artifacts.dataframe(
    filename.replace(".csv", "")
)

        df.to_csv(
            path,
            index=False
        )

        logger.info(
            f"Saved dataframe -> {path}"
        )

    def save_json(
        self,
        data: dict,
        filename: str
    ):

        path = self.experiment_dir / filename

        with open(path, "w") as f:

            json.dump(
                data,
                f,
                indent=4,
                default=str
            )

        logger.info(
            f"Saved json -> {path}"
        )

    def save_model(
        self,
        model,
        filename: str = "model.joblib"
    ):

        path = self.artifacts.model(filename.replace(".joblib", ""))

        joblib.dump(
            model,
            path
        )

        logger.info(
            f"Saved model -> {path}"
        )

    def save_figure(
        self,
        fig,
        filename: str
    ):

        path = self.plots_dir / filename

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        logger.info(
            f"Saved figure -> {path}"
        )

    def save_text(
        self,
        text: str,
        filename: str
    ):

        path = self.experiment_dir / filename

        with open(path, "w") as f:

            f.write(text)

        logger.info(
            f"Saved text -> {path}"
        )

    def path(self):

        return self.experiment_dir

    def summary(self):

        logger.info("=" * 70)
        logger.info("Experiment Summary")
        logger.info("=" * 70)

        logger.info(f"Directory : {self.experiment_dir}")
        logger.info(f"Plots     : {self.plots_dir}")
        logger.info(f"Models    : {self.models_dir}")
        logger.info(f"Data      : {self.data_dir}")

        logger.info("=" * 70)