from __future__ import annotations

import logging
from dataclasses import asdict

from ml.inference.loader import ArtifactLoader
from ml.inference.preprocess import InferencePreprocessor
from ml.inference.schemas import PredictionResponse, PredictionRequest

logger = logging.getLogger(__name__)
class StrokePredictor:

    def __init__(self):

        logger.info("Loading inference artifacts...")

        artifacts = ArtifactLoader().load_all()

        self.model = artifacts["model"]

        self.preprocessor = artifacts["preprocessor"]

        self.threshold = artifacts["threshold"]

        self.feature_names = artifacts["feature_names"]

        self.model_info = artifacts["model_info"]

        self.processor = InferencePreprocessor()

    def predict(
    self,
    patient: PredictionRequest,
    ) -> PredictionResponse:

        patient = asdict(patient)

        logger.info("Running inference...")

        df = self.processor.preprocess(
            patient
        )

        X = self.preprocessor.transform(
            df
        )

        probability = float(

            self.model.predict_proba(X)[0][1]

        )

        prediction = int(

            probability >= self.threshold

        )

        if probability >= 0.85:

            risk = "Very High"

        elif probability >= 0.60:

            risk = "High"

        elif probability >= 0.30:

            risk = "Moderate"

        else:

            risk = "Low"

        response = PredictionResponse(

            prediction=prediction,

            probability=round(
                probability,
                4,
            ),

            threshold=self.threshold,

            risk=risk,

            model=self.model_info.get(
                "model",
                "logistic_regression",
            ),

            version=self.model_info.get(
                "version",
                "v1.0",
            ),

        )

        logger.info("Inference completed.")

        return response

    def predict_dict(
        self,
        patient: dict,
    ):

        """
        Useful for APIs.
        """

        return asdict(

            self.predict(patient)

        )

if __name__ == "__main__":

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s | %(levelname)s | %(message)s",

    )

    patient = {

        "gender": "Male",

        "age": 67,

        "hypertension": 1,

        "heart_disease": 1,

        "ever_married": "Yes",

        "work_type": "Private",

        "Residence_type": "Urban",

        "avg_glucose_level": 228.69,

        "bmi": 36.6,

        "smoking_status": "formerly smoked",

    }

    predictor = StrokePredictor()

    result = predictor.predict(patient)

    print()

    print(result)

    print()

    print(predictor.predict_dict(patient))