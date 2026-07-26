from ml.data.load_data import load_config, load_dataset
from ml.features.split import split_dataset
from ml.features.preprocess import DataPreprocessor
from ml.models.train import ModelTrainer
from ml.models.evaluate import ModelEvaluator
from ml.models.experiment import ExperimentRunner
from ml.validation.cross_validation import CrossValidator
from ml.models.registry import get_model_registry
from ml.validation.threshold import ThresholdOptimizer

config = load_config("configs/config.yaml")

df = load_dataset(config)

X_train, X_test, y_train, y_test = split_dataset(df, config)

preprocessor = DataPreprocessor()

preprocessor.detect_features(X_train)

preprocessor.build_numeric_pipeline()

preprocessor.build_categorical_pipeline()

preprocessor.build_preprocessor()

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

preprocessor.save_preprocessor()

feature_names = preprocessor.get_feature_names()

trainer = ModelTrainer()

trainer.train(
    model_name="logistic_regression",
    X_train=X_train_processed,
    y_train=y_train
)

trainer.save_model()

predictions = trainer.predict(X_test_processed)

probabilities = trainer.predict_proba(X_test_processed)

print()

print("=" * 60)
print("Processed Dataset Summary")
print("=" * 60)

print("Train Shape :", X_train_processed.shape)
print("Test Shape  :", X_test_processed.shape)
print("Features    :", len(feature_names))

print()

print("First 10 Features")

for feature in feature_names[:10]:
    print(feature)

print("=" * 60)
print("Prediction Check")
print("=" * 60)

print(predictions[:10])

print()

print(probabilities[:5])

evaluator = ModelEvaluator()

results = evaluator.evaluate(
    trainer,
    X_test_processed,
    y_test
)

print()

print("=" * 60)
print("Evaluation Metrics")
print("=" * 60)

for metric, value in results["metrics"].items():
    print(f"{metric:15}: {value:.4f}")

print()

print("=" * 60)
print("Confusion Matrix")
print("=" * 60)

print(results["confusion_matrix"])

runner = ExperimentRunner()

leaderboard = runner.run(

    X_train_processed,

    y_train,

    X_test_processed,

    y_test

)

print()

print("=" * 70)
print("MODEL LEADERBOARD")
print("=" * 70)

print(leaderboard)

registry = get_model_registry()

cv = CrossValidator()

results = cv.evaluate(
    registry["logistic_regression"],
    X_train_processed,
    y_train
)

print()

print("=" * 60)
print("5-Fold Cross Validation")
print("=" * 60)

print(results)

optimizer = ThresholdOptimizer()

threshold_results = optimizer.evaluate(
    trainer.model,
    X_test_processed,
    y_test
)

print(threshold_results)

best = optimizer.best_threshold(
    threshold_results,
    metric="f1_score"
)

print()

print("=" * 60)
print("Best Threshold")
print("=" * 60)

print(best)