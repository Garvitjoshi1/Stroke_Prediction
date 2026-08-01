import joblib

obj = joblib.load("artifacts/models/logistic_regression.joblib")

print(type(obj))

if isinstance(obj, dict):
    print(obj.keys())