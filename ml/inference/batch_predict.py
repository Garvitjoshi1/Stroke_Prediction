from __future__ import annotations

import logging

import pandas as pd

from ml.inference.predictor import StrokePredictor
from ml.inference.schemas import PredictionRequest


logger = logging.getLogger(__name__)


class BatchPredictor:
    
    def __init__(self):

        self.predictor = StrokePredictor()

    def predict_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Running batch prediction..."
        )

        predictions = []

        for _, row in dataframe.iterrows():

            request = PredictionRequest(

                gender=row["gender"],

                age=row["age"],

                hypertension=row["hypertension"],

                heart_disease=row["heart_disease"],

                ever_married=row["ever_married"],

                work_type=row["work_type"],

                Residence_type=row["Residence_type"],

                avg_glucose_level=row["avg_glucose_level"],

                bmi=row["bmi"],

                smoking_status=row["smoking_status"],

            )

            result = self.predictor.predict(request)

            predictions.append({

                **row.to_dict(),

                "prediction": result.prediction,

                "probability": result.probability,

                "risk": result.risk,

            })

        logger.info(

            "Batch prediction completed."

        )

        return pd.DataFrame(predictions)

    def predict_csv(
        self,
        csv_path: str,
    ) -> pd.DataFrame:

        logger.info(

            "Loading CSV..."

        )

        df = pd.read_csv(csv_path)

        return self.predict_dataframe(df)

    def save_predictions(
        self,
        dataframe: pd.DataFrame,
        output_path: str,
    ):

        dataframe.to_csv(

            output_path,

            index=False,

        )

        logger.info(

            "Predictions saved to %s",

            output_path,

        )

if __name__ == "__main__":

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s | %(levelname)s | %(message)s",

    )

    sample = pd.DataFrame([

        {

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

        },

        {

            "gender": "Female",

            "age": 42,

            "hypertension": 0,

            "heart_disease": 0,

            "ever_married": "Yes",

            "work_type": "Private",

            "Residence_type": "Urban",

            "avg_glucose_level": 91.2,

            "bmi": 24.8,

            "smoking_status": "never smoked",

        },

    ])

    predictor = BatchPredictor()

    results = predictor.predict_dataframe(sample)

    print()

    print(results)