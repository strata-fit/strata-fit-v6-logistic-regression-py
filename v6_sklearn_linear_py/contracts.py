from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LogisticRegressionPartialInput(BaseModel):
    model_attributes: Dict[str, Any]
    predictors: List[str] = Field(min_length=1)
    outcome: str
    n_local_iterations: int = 1
    model_class: Any = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)


class LogisticRegressionPartialOutput(BaseModel):
    model_attributes: Dict[str, Any]
    size: int


class ComputeLossInput(BaseModel):
    model_attributes: Dict[str, Any]
    predictors: List[str] = Field(min_length=1)
    outcome: str
    model_class: Any = None


class ComputeLossOutput(BaseModel):
    loss: float
    size: int


class RunValidationInput(BaseModel):
    parameters: Any
    classes: Optional[List[Any]] = None
    predictors: List[str] = Field(min_length=1)
    outcome: str
    model_class: Any = None


class RunValidationOutput(BaseModel):
    score: float
    confusion_matrix: Optional[List[List[int]]] = None


class MasterFlowerInput(BaseModel):
    org_ids: List[int] = Field(min_length=1)
    predictors: List[str] = Field(min_length=1)
    outcome: str
    classes: Optional[List[Any]] = None
    database_label: str = "default"
    num_rounds: int = 5
    n_local_epochs: int = 1
    strategy_name: str = "fedavg"
    strategy_kwargs: Dict[str, Any] = Field(default_factory=dict)
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)


class MasterFlowerOutput(BaseModel):
    model_attributes: Dict[str, Any]
    history: List[Dict[str, Any]]
