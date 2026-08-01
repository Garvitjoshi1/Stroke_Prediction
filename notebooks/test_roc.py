import joblib

from ml.visualization.roc import ROCPlotter

model = joblib.load(
    "artifacts/models/logistic_regression.joblib"
)

X_test = joblib.load(
    "artifacts/datasets/X_test_processed.pkl"
)

y_test = joblib.load(
    "artifacts/datasets/y_test.pkl"
)

ROCPlotter().plot_and_save(
    model,
    X_test,
    y_test,
)