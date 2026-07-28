from dataclasses import dataclass


@dataclass
class ExperimentResult:

    sampler: str

    accuracy: float

    precision: float

    recall: float

    f1: float

    roc_auc: float

    pr_auc: float

    training_time: float