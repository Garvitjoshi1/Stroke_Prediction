from __future__ import annotations

from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):

    gender: str = Field(
        ...,
        examples=["Male"]
    )

    age: float = Field(
        ...,
        ge=0,
        le=120,
        examples=[67]
    )

    hypertension: int = Field(
        ...,
        ge=0,
        le=1,
        examples=[1]
    )

    heart_disease: int = Field(
        ...,
        ge=0,
        le=1,
        examples=[1]
    )

    ever_married: str = Field(
        ...,
        examples=["Yes"]
    )

    work_type: str = Field(
        ...,
        examples=["Private"]
    )

    Residence_type: str = Field(
        ...,
        examples=["Urban"]
    )

    avg_glucose_level: float = Field(
        ...,
        ge=0,
        examples=[228.69]
    )

    bmi: float = Field(
        ...,
        ge=0,
        examples=[36.6]
    )

    smoking_status: str = Field(
        ...,
        examples=["formerly smoked"]
    )

class PredictionResponse(BaseModel):

    prediction: int

    probability: float

    threshold: float

    risk: str

    model: str

    version: str

class HealthResponse(BaseModel):

    status: str

    model_loaded: bool

    version: str

from pydantic import BaseModel

class PatientInput(BaseModel):

    age: float

    gender: str

    hypertension: int

    heart_disease: int

    ever_married: str

    work_type: str

    Residence_type: str

    avg_glucose_level: float

    bmi: float

    smoking_status: str