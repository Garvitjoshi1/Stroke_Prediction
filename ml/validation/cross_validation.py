import logging
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class CrossValidator:

    def __init__(
        self,
        n_splits=5,
        shuffle=True,
        random_state=42
    ):

        self.cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

    def evaluate(
        self,
        model,
        X,
        y
    ):

        logger.info("Running Stratified K-Fold Validation...")

        scoring = {

            "accuracy": "accuracy",

            "precision": "precision",

            "recall": "recall",

            "f1": "f1",

            "roc_auc": "roc_auc"

        }

        scores = cross_validate(

            estimator=model,

            X=X,

            y=y,

            cv=self.cv,

            scoring=scoring,

            n_jobs=-1,

            return_train_score=False

        )

        results = pd.DataFrame({

            "Metric": [

                "Accuracy",

                "Precision",

                "Recall",

                "F1",

                "ROC_AUC"

            ],

            "Mean": [

                scores["test_accuracy"].mean(),

                scores["test_precision"].mean(),

                scores["test_recall"].mean(),

                scores["test_f1"].mean(),

                scores["test_roc_auc"].mean()

            ],

            "Std": [

                scores["test_accuracy"].std(),

                scores["test_precision"].std(),

                scores["test_recall"].std(),

                scores["test_f1"].std(),

                scores["test_roc_auc"].std()

            ]

        })

        logger.info("Cross Validation Completed.")

        return results