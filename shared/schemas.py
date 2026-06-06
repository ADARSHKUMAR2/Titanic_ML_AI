from typing import List, Literal
from pydantic import BaseModel, Field

class PassengerFeature(BaseModel):
    PassengerId: int = Field(description="The unique identifier of the passenger.")
    ai_survival_probability: float = Field(
        description="Evaluated survival probability between 0.0 (certain death) and 1.0 (certain survival).",
        ge=0.0,
        le=1.0
    )
    estimated_social_tier: Literal["High", "Medium", "Low"] = Field(
        description="Socio-economic classification determined by title, class, and fare metrics."
    )

class TitanicBatchResponse(BaseModel):
    passengers: List[PassengerFeature] = Field(description="List of evaluated passenger features.")