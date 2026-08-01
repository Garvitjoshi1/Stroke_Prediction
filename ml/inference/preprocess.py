from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


class InferencePreprocessor:
    
    REQUIRED_COLUMNS = [

        "gender",

        "age",

        "hypertension",

        "heart_disease",

        "ever_married",

        "work_type",

        "Residence_type",

        "avg_glucose_level",

        "bmi",

        "smoking_status",

    ]

    NUMERIC_COLUMNS = [

        "age",

        "hypertension",

        "heart_disease",

        "avg_glucose_level",

        "bmi",

    ]

    STRING_COLUMNS = [

        "gender",

        "ever_married",

        "work_type",

        "Residence_type",

        "smoking_status",

    ]

    def validate_input(
        self,
        patient: Dict,
    ):

        logger.info("Validating input...")

        missing = []

        for column in self.REQUIRED_COLUMNS:

            if column not in patient:

                missing.append(column)

        if len(missing):

            raise ValueError(

                f"Missing required fields: {missing}"

            )

        logger.info("Validation successful.")

    def create_dataframe(
        self,
        patient: Dict,
    ) -> pd.DataFrame:

        logger.info("Creating dataframe...")

        df = pd.DataFrame([patient])

        return df

    def enforce_datatypes(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info("Applying datatypes...")

        for column in self.NUMERIC_COLUMNS:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        for column in self.STRING_COLUMNS:

            df[column] = df[column].astype(str)

        return df

    def reorder_columns(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info("Reordering columns...")

        return df[self.REQUIRED_COLUMNS]

    def preprocess(
        self,
        patient: Dict,
    ) -> pd.DataFrame:

        self.validate_input(patient)

        df = self.create_dataframe(patient)

        df = self.enforce_datatypes(df)

        df = self.reorder_columns(df)

        logger.info(
            "Inference dataframe ready."
        )

        return df


if __name__ == "__main__":

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s | %(levelname)s | %(message)s",

    )

    sample = {

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

    processor = InferencePreprocessor()

    df = processor.preprocess(sample)

    print()

    print(df)

    print()

    print(df.dtypes)