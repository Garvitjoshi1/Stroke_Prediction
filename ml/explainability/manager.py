from __future__ import annotations

import logging
from pathlib import Path

from ml.explainability.global_explainer import GlobalExplainer
from ml.explainability.summary import SummaryPlotter
from ml.explainability.beeswarm import BeeswarmPlotter
from ml.explainability.waterfall import WaterfallPlotter
from ml.explainability.local_explainer import LocalExplainer
from ml.explainability.dependence import DependencePlotter

logger = logging.getLogger(__name__)


class ExplainabilityManager:

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

        self.global_plotter = GlobalExplainer(
            model=model,
            output_dir=output_dir,
        )

        self.summary_plotter = SummaryPlotter(
            model=model,
            output_dir=output_dir,
        )

        self.beeswarm_plotter = BeeswarmPlotter(
            model=model,
            output_dir=output_dir,
        )

        self.waterfall_plotter = WaterfallPlotter(
            model=model,
            output_dir=output_dir,
        )

        self.local_explainer = LocalExplainer(
            model=model,
            output_dir=output_dir,
        )

        self.dependence_plotter = DependencePlotter(
            model=model,
            output_dir=output_dir,
        )

    def generate_all(
        self,
        X,
        feature_names,
        sample_index: int = 0,
        dependence_features=None,
    ):

        logger.info("=" * 70)
        logger.info("Generating Explainability Artifacts")
        logger.info("=" * 70)

        self.global_plotter.plot(
            X,
            feature_names,
        )

        self.summary_plotter.plot(
            X,
            feature_names,
        )

        self.beeswarm_plotter.plot(
            X,
            feature_names,
        )

        self.waterfall_plotter.plot(
            X,
            sample_index=sample_index,
        )

        explanation = self.local_explainer.explain(
            X,
            feature_names,
            sample_index=sample_index,
            generate_plot=False,
        )

        if dependence_features is None:

            dependence_features = feature_names[:5]

        for feature in dependence_features:

            try:

                self.dependence_plotter.plot(
                    X,
                    feature_names,
                    feature,
                )

            except Exception as e:

                logger.warning(
                    "Failed dependence plot for %s : %s",
                    feature,
                    e,
                )

        logger.info("=" * 70)
        logger.info("Explainability completed.")
        logger.info("=" * 70)

        return explanation