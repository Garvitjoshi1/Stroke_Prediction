from __future__ import annotations

from dataclasses import dataclass

@dataclass
class PredictionRequest:

    gender: str

    age: float

    hypertension: int

    heart_disease: int

    ever_married: str

    work_type: str

    Residence_type: str

    avg_glucose_level: float

    bmi: float

    smoking_status: str

@dataclass
class PredictionResponse:

    prediction: int

    probability: float

    threshold: float

    risk: str

    model: str

    version: str

if __name__ == "__main__":

    request = PredictionRequest(

        gender="Male",

        age=67,

        hypertension=1,

        heart_disease=1,

        ever_married="Yes",

        work_type="Private",

        Residence_type="Urban",

        avg_glucose_level=228.69,

        bmi=36.6,

        smoking_status="formerly smoked",

    )

    print(request)

    print()

    response = PredictionResponse(

        prediction=1,

        probability=0.9123,

        threshold=0.85,

        risk="Very High",

        model="logistic_regression",

        version="v1.0",

    )

    print(response)