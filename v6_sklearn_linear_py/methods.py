from typing import Any, Dict, Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression

from v6_federated_core import MethodContext, MethodRegistry, MethodSpec

from .contracts import (
    ComputeLossInput,
    ComputeLossOutput,
    LogisticRegressionPartialInput,
    LogisticRegressionPartialOutput,
    MasterFlowerInput,
    MasterFlowerOutput,
    RunValidationInput,
    RunValidationOutput,
)
from .flower_bridge import _master_flower_core
from .partials import _compute_loss_partial, _logistic_regression_partial, _run_validation


def _get_dataframe(context: MethodContext) -> pd.DataFrame:
    df = context.meta.get("df")
    if df is None:
        raise RuntimeError("Method context is missing the dataframe")
    return df


def _get_client(context: MethodContext):
    client = context.meta.get("client")
    if client is None:
        raise RuntimeError("Method context is missing the AlgorithmClient")
    return client


def logistic_regression_partial_handler(
    data: LogisticRegressionPartialInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for logistic_regression_partial")
    df = _get_dataframe(context)
    model_class = data.model_class or LogisticRegression
    return _logistic_regression_partial(
        df,
        data.model_attributes,
        data.predictors,
        data.outcome,
        data.n_local_iterations,
        model_class=model_class,
        **data.model_kwargs,
    )


def compute_loss_partial_handler(
    data: ComputeLossInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for compute_loss_partial")
    df = _get_dataframe(context)
    model_class = data.model_class or LogisticRegression
    return _compute_loss_partial(
        df,
        data.model_attributes,
        data.predictors,
        data.outcome,
        model_class=model_class,
    )


def run_validation_handler(
    data: RunValidationInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for run_validation")
    df = _get_dataframe(context)
    model_class = data.model_class or LogisticRegression
    return _run_validation(
        df,
        data.parameters,
        data.classes,
        data.predictors,
        data.outcome,
        model_class=model_class,
    )


def master_flower_handler(
    data: MasterFlowerInput,
    context: Optional[MethodContext] = None,
) -> Dict[str, Any]:
    if context is None:
        raise RuntimeError("Method context is required for master_flower")
    client = _get_client(context)
    return _master_flower_core(
        client=client,
        org_ids=data.org_ids,
        predictors=data.predictors,
        outcome=data.outcome,
        classes=data.classes,
        database_label=data.database_label,
        num_rounds=data.num_rounds,
        n_local_epochs=data.n_local_epochs,
        strategy_name=data.strategy_name,
        strategy_kwargs=data.strategy_kwargs,
        model_kwargs=data.model_kwargs,
    )


METHOD_REGISTRY = MethodRegistry(
    [
        MethodSpec(
            name="logistic_regression_partial",
            input_model=LogisticRegressionPartialInput,
            output_model=LogisticRegressionPartialOutput,
            handler=logistic_regression_partial_handler,
        ),
        MethodSpec(
            name="compute_loss_partial",
            input_model=ComputeLossInput,
            output_model=ComputeLossOutput,
            handler=compute_loss_partial_handler,
        ),
        MethodSpec(
            name="run_validation",
            input_model=RunValidationInput,
            output_model=RunValidationOutput,
            handler=run_validation_handler,
        ),
        MethodSpec(
            name="master_flower",
            input_model=MasterFlowerInput,
            output_model=MasterFlowerOutput,
            handler=master_flower_handler,
        ),
    ]
)
