import logging
import pandas as pd

from ml.imbalance.samplers import SamplerFactory
from ml.models.train import ModelTrainer
from ml.models.evaluate import ModelEvaluator

logger = logging.getLogger(__name__)


class ImbalanceExperiment:
    def __init__(self):

        self.samplers = SamplerFactory.get_samplers()

    def run(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        model_name="logistic_regression"
    ):

        leaderboard = []

        for sampler_name, sampler in self.samplers.items():

            logger.info("=" * 70)
            logger.info(f"Running Sampler : {sampler_name}")
            logger.info("=" * 70)

            if sampler is None:

                X_resampled = X_train
                y_resampled = y_train

            else:

                X_resampled, y_resampled = sampler.fit_resample(
                    X_train,
                    y_train
                )

            logger.info(
                f"Resampled Dataset Shape : {X_resampled.shape}"
            )

            logger.info(
                f"Training Samples : {len(y_resampled)}"
            )

            trainer = ModelTrainer()

            trainer.train(
                model_name,
                X_resampled,
                y_resampled
            )

            evaluator = ModelEvaluator()

            results = evaluator.evaluate(
                trainer.model,
                X_test,
                y_test
            )

            metrics = results["metrics"]

            leaderboard.append({

                "sampler": sampler_name,

                "accuracy": metrics["accuracy"],

                "precision": metrics["precision"],

                "recall": metrics["recall"],

                "f1_score": metrics["f1_score"],

                "roc_auc": metrics["roc_auc"],

                "pr_auc": metrics["pr_auc"],

                "training_time": trainer.training_time

            })

        leaderboard = pd.DataFrame(leaderboard)

        leaderboard = leaderboard.sort_values(

            by="roc_auc",

            ascending=False

        ).reset_index(drop=True)

        logger.info("=" * 70)
        logger.info("Imbalance experiments completed.")
        logger.info("=" * 70)

        return leaderboard