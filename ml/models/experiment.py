import logging
import pandas as pd

from ml.models.registry import get_model_registry
from ml.models.train import ModelTrainer
from ml.models.evaluate import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class ExperimentRunner:

    def __init__(self):

        self.results = []

    def run(

        self,

        X_train,

        y_train,

        X_test,

        y_test

    ):

        registry = get_model_registry()

        for model_name in registry.keys():

            logger.info("=" * 60)
            logger.info(f"Training {model_name}")
            logger.info("=" * 60)

            trainer = ModelTrainer()

            trainer.train(
                model_name,
                X_train,
                y_train
            )

            trainer.save_model()

            evaluator = ModelEvaluator()

            result = evaluator.evaluate(
                trainer,
                X_test,
                y_test
            )

            metrics = result["metrics"]

            metrics["model"] = model_name

            metrics["training_time"] = trainer.training_time

            self.results.append(metrics)

        leaderboard = pd.DataFrame(self.results)

        leaderboard = leaderboard.sort_values(
            by="roc_auc",
            ascending=False
        )

        return leaderboard.reset_index(drop=True)