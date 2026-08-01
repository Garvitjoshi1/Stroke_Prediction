from ml.visualization.base import BasePlotter

plotter = BasePlotter()

fig, ax = plotter.create_subplots()

ax.plot([1, 2, 3], [4, 2, 6])

plotter.style_axis(
    ax,
    title="Test Figure",
    xlabel="X",
    ylabel="Y",
)

plotter.finish(
    fig,
    "test.png",
)

import joblib

from ml.visualization.pr import PRCurvePlotter

model = joblib.load(
    "artifacts/models/logistic_regression.joblib"
)

X_test = joblib.load(
    "artifacts/datasets/X_test_processed.pkl"
)

y_test = joblib.load(
    "artifacts/datasets/y_test.pkl"
)

PRCurvePlotter().plot_and_save(
    model,
    X_test,
    y_test
)

import joblib

from ml.visualization.confusion import ConfusionMatrixPlotter

model = joblib.load(
    "artifacts/models/logistic_regression.joblib"
)

X_test = joblib.load(
    "artifacts/datasets/X_test_processed.pkl"
)

y_test = joblib.load(
    "artifacts/datasets/y_test.pkl"
)

ConfusionMatrixPlotter().plot_and_save(
    model,
    X_test,
    y_test,
    normalize=True,
)

import joblib

from ml.visualization.threshold import ThresholdPlotter

model = joblib.load(
    "artifacts/models/logistic_regression.joblib"
)

X_test = joblib.load(
    "artifacts/datasets/X_test_processed.pkl"
)

y_test = joblib.load(
    "artifacts/datasets/y_test.pkl"
)

ThresholdPlotter().plot_and_save(
    model,
    X_test,
    y_test,
)

import joblib

from ml.visualization.calibration import CalibrationPlotter

model = joblib.load(
    "artifacts/models/logistic_regression.joblib"
)

X_test = joblib.load(
    "artifacts/datasets/X_test_processed.pkl"
)

y_test = joblib.load(
    "artifacts/datasets/y_test.pkl"
)

CalibrationPlotter().plot_and_save(
    model,
    X_test,
    y_test,
)

import joblib

from ml.visualization.importance import FeatureImportancePlotter

model = joblib.load(
    "artifacts/models/logistic_regression.joblib"
)

X_test = joblib.load(
    "artifacts/datasets/X_test_processed.pkl"
)

y_test = joblib.load(
    "artifacts/datasets/y_test.pkl"
)

feature_names = joblib.load(
    "artifacts/preprocessors/feature_names.joblib"
)

FeatureImportancePlotter().plot_and_save(
    model,
    X_test,
    y_test,
    feature_names,
)