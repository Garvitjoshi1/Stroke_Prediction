from __future__ import annotations

from fastapi import APIRouter, Depends

from apps.api.dependencies import get_predictor
from apps.api.schemas import HealthResponse
from apps.api.config import settings

from ml.inference.predictor import StrokePredictor


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get(
    "",
    response_model=HealthResponse,
    summary="Health Check",
)
def health_check(
    predictor: StrokePredictor = Depends(
        get_predictor
    ),
):

    return HealthResponse(

        status="healthy",

        model_loaded=predictor is not None,

        version=settings.VERSION,

    )