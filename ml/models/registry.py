from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier


def get_model_registry():

    models = {

        "logistic_regression": LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ),

        "decision_tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),

        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),

        "gradient_boosting": GradientBoostingClassifier(
            random_state=42
        )

    }

    return models