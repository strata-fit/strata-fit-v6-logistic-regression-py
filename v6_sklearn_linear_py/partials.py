import warnings

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from typing import Any, Dict, List, Optional, Type, Union
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client, data
from v6_federated_core import MethodContext, dispatch_registered_method, to_v6_result

from v6_sklearn_linear_py.helper import (
    coordinate_task,
    export_model,
    filter_model_init_kwargs,
    initialize_model,
    resolve_linear_model_class,
)

MODEL_ATTRIBUTE_KEYS = ["coef_", "intercept_", "classes_"]

@data(1)
def logistic_regression_partial(
    df: pd.DataFrame, 
    model_attributes: Dict[str, List[float]], 
    predictors: List[str], 
    outcome: str,
    n_local_iterations: int = 1,
    model_class: Union[str, Type[LogisticRegression]] = LogisticRegression,
    **model_kwargs
) -> Dict[str, any]:
    from v6_sklearn_linear_py.methods import METHOD_REGISTRY

    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "logistic_regression_partial",
        {
            "model_attributes": model_attributes,
            "predictors": predictors,
            "outcome": outcome,
            "n_local_iterations": n_local_iterations,
            "model_class": model_class,
            "model_kwargs": model_kwargs,
        },
        context=MethodContext(method="logistic_regression_partial", meta={"df": df}),
    )
    return to_v6_result(envelope)



def _logistic_regression_partial(
    df: pd.DataFrame, 
    model_attributes: Dict[str, List[float]], 
    predictors: List[str], 
    outcome: str,
    n_local_iterations: int = 1,
    model_class: Union[str, Type[LogisticRegression]] = LogisticRegression,
    **model_kwargs
) -> Dict[str, any]:
    """
    Fits a linear-model estimator (defaults to LogisticRegression) on the local dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Local data frame.
    model_attributes : Dict[str, List[float]]
        Logistic regression model attributes (weights, intercepts).
    predictors : List[str]
        List of predictor variable names.
    outcome : str
        Outcome variable name.
    model_class : Union[str, Type[BaseEstimator]]
        Class (or import path) from sklearn.linear_model to instantiate.

    Returns
    -------
    Dict[str, any]
        Attributes of locally trained logistic regression model and local dataset size.
    """
    # Drop rows with NaNs
    df = df.dropna(how='any')

    # Get features and outcomes
    X = df[predictors].values
    y = df[outcome].values

    # Create local LogisticRegression estimator object
    base_kwargs = dict(max_iter=n_local_iterations, warm_start=True)
    model_cls = resolve_linear_model_class(model_class)
    # Merge base + extras (penalty, solver, l1_ratio, C, etc.) and drop unsupported keys
    candidate_kwargs = {**base_kwargs, **{k: v for k, v in model_kwargs.items() if v is not None}}
    init_kwargs = filter_model_init_kwargs(model_cls, candidate_kwargs)

    model = initialize_model(model_cls, model_attributes=model_attributes, **init_kwargs)
    
    # Ignore convergence failure due to low local epochs
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X, y)
        info('Training round finished')
    attribute_keys = [k for k in MODEL_ATTRIBUTE_KEYS if hasattr(model, k)]
    model_attributes = export_model(model, attribute_keys=attribute_keys)

    return {
        'model_attributes': model_attributes,
        'size': X.shape[0]
    }


@data(1)
def compute_loss_partial(
    df: pd.DataFrame, 
    model_attributes: Dict[str, list], 
    predictors: List[str], 
    outcome: str,
    model_class: Union[str, Type[LogisticRegression]] = LogisticRegression,
) -> Dict[str, Any]:
    from v6_sklearn_linear_py.methods import METHOD_REGISTRY

    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "compute_loss_partial",
        {
            "model_attributes": model_attributes,
            "predictors": predictors,
            "outcome": outcome,
            "model_class": model_class,
        },
        context=MethodContext(method="compute_loss_partial", meta={"df": df}),
    )
    return to_v6_result(envelope)

def _compute_loss_partial(
    df: pd.DataFrame, 
    model_attributes: Dict[str, list], 
    predictors: List[str], 
    outcome: str,
    model_class: Union[str, Type[LogisticRegression]] = LogisticRegression,
) -> Dict[str, Any]:
    """
    Computes model loss on the local dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Local data frame.
    model_attributes : Dict[str, list]
        Attributes of the logistic regression model.
    predictors : List[str]
        Predictor variables.
    outcome : str
        Outcome variable.
    model_class : Union[str, Type[BaseEstimator]]
        Class (or import path) from sklearn.linear_model to instantiate.

    Returns
    -------
    Dict[str, Any]
        Local loss and dataset size.
    """
    # Drop rows with NaNs
    df = df.dropna(how='any')

    # Get features and outcomes
    X = df[predictors].values
    y = df[outcome].values

    # Initialize local model instance
    model_cls = resolve_linear_model_class(model_class)
    model = initialize_model(model_cls, model_attributes)

    # Compute a classification loss when probabilities are available, otherwise MSE.
    if isinstance(model, ClassifierMixin) and hasattr(model, "predict_proba"):
        labels = getattr(model, "classes_", None)
        loss = log_loss(y, model.predict_proba(X), labels=labels)
    else:
        y_pred = model.predict(X)
        loss = float(np.mean((y - y_pred) ** 2))

    return {
        'loss': loss,
        'size': X.shape[0]
    }


@data(1)
def run_validation(
    df: pd.DataFrame, 
    parameters: Union[List[np.ndarray], Dict[str, Any]], 
    classes: Optional[List[str]], 
    predictors: List[str], 
    outcome: str,
    model_class: Union[str, Type[LogisticRegression]] = LogisticRegression,
) -> Dict[str, Any]:
    from v6_sklearn_linear_py.methods import METHOD_REGISTRY

    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "run_validation",
        {
            "parameters": parameters,
            "classes": classes,
            "predictors": predictors,
            "outcome": outcome,
            "model_class": model_class,
        },
        context=MethodContext(method="run_validation", meta={"df": df}),
    )
    return to_v6_result(envelope)

def _run_validation(
    df: pd.DataFrame, 
    parameters: Union[List[np.ndarray], Dict[str, Any]], 
    classes: Optional[List[str]], 
    predictors: List[str], 
    outcome: str,
    model_class: Union[str, Type[LogisticRegression]] = LogisticRegression,
) -> Dict[str, Any]:
    """
    Validates a linear-model estimator on the local dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Local data frame for validation.
    parameters : Union[List[np.ndarray], Dict[str, Any]]
        Model parameters for validation. If a list/tuple is provided, the legacy
        order is [intercept, coef]. If a dict is provided, keys are used directly.
    classes : List[str], optional
        List of class labels (provide for classifiers that require classes_).
    predictors : List[str]
        Predictor variables for validation.
    outcome : str
        Outcome variable for validation.
    model_class : Union[str, Type[BaseEstimator]]
        Class (or import path) from sklearn.linear_model to instantiate.

    Returns
    -------
    Dict[str, Any]
        Performance metrics including model accuracy and confusion matrix.
    """
    # Drop rows with NaNs
    df = df.dropna(how='any')

    # Get features and outcomes
    X = df[predictors].values
    y = df[outcome].values

    # Initialize estimator
    # Build attributes dict from legacy list/tuple or explicit dict
    if isinstance(parameters, dict):
        model_attributes = {k: np.array(v) for k, v in parameters.items()}
    elif isinstance(parameters, (list, tuple)):
        model_attributes = {}
        if len(parameters) >= 1:
            # legacy ordering: [intercept, coef]
            model_attributes["intercept_"] = np.array(parameters[0])
        if len(parameters) >= 2:
            model_attributes["coef_"] = np.array(parameters[1])
    else:
        raise TypeError("parameters must be a dict or a list/tuple")

    model_cls = resolve_linear_model_class(model_class)
    is_classification = issubclass(model_cls, ClassifierMixin)

    if classes is not None and is_classification:
        model_attributes["classes_"] = np.array(classes)
    if not is_classification:
        model_attributes.pop("classes_", None)

    model = initialize_model(model_cls, model_attributes)

    # Compute score (accuracy for classifiers, R^2 for regressors)
    score = model.score(X, y)

    result: Dict[str, Any] = {'score': score}

    # Confusion matrix is only defined for classification outputs.
    if is_classification:
        labels = classes if classes is not None else getattr(model, "classes_", None)
        if labels is not None:
            labels = list(labels)
    else:
        labels = None

    if labels:
        result['confusion_matrix'] = confusion_matrix(
            y, model.predict(X), labels=labels
        ).tolist()

    return result
