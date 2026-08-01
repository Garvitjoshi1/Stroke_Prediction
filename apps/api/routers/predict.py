from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from apps.api.dependencies import get_predictor
from apps.api.schemas import (
    PredictionRequest,
    PredictionResponse,
)

from ml.inference.predictor import StrokePredictor
from ml.inference.schemas import (
    PredictionRequest as MLPredictionRequest,
)


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Predict Stroke Risk",
)
def predict(
    request: PredictionRequest,
    predictor: StrokePredictor = Depends(get_predictor),
):

    try:

        ml_request = MLPredictionRequest(

            gender=request.gender,

            age=request.age,

            hypertension=request.hypertension,

            heart_disease=request.heart_disease,

            ever_married=request.ever_married,

            work_type=request.work_type,

            Residence_type=request.Residence_type,

            avg_glucose_level=request.avg_glucose_level,

            bmi=request.bmi,

            smoking_status=request.smoking_status,

        )

        result = predictor.predict(
            ml_request
        )

        return PredictionResponse(

            prediction=result.prediction,

            probability=result.probability,

            threshold=result.threshold,

            risk=result.risk,

            model=result.model,

            version=result.version,

        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e),

        )