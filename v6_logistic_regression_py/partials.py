import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from typing import Any, Dict, List
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client, data

from v6_logistic_regression_py.helper import (
    aggregate,
    coordinate_task,
    export_model,
    initialize_model
)

MODEL_ATTRIBUTE_KEYS = ["coef_", "intercept_", "classes_"]

@data(1)
def logistic_regression_partial(
    df: pd.DataFrame, 
    model_attributes: Dict[str, List[float]], 
    predictors: List[str], 
    outcome: str,
    n_local_iterations: int = 1,
    **model_kwargs
) -> Dict[str, any]:
    """
    Fits logistic regression model on local dataset.

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
    base_kwargs = dict(
        max_iter=n_local_iterations,
        warm_start=True,
    )
    # Merge base + extras (penalty, solver, l1_ratio, C, etc.)
    full_kwargs = {**base_kwargs, **{k: v for k, v in model_kwargs.items() if v is not None}}
    model = initialize_model(LogisticRegression, model_attributes=model_attributes, **model_kwargs)
    
    # Ignore convergence failure due to low local epochs
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X, y)
        info('Training round finished')
    
    model_attributes = export_model(model, attribute_keys=MODEL_ATTRIBUTE_KEYS)

    return {
        'model_attributes': model_attributes,
        'size': X.shape[0]
    }


@data(1)
def compute_loss_partial(
    df: pd.DataFrame, 
    model_attributes: Dict[str, list], 
    predictors: List[str], 
    outcome: str
) -> Dict[str, Any]:
    """
    Computes logistic regression model loss on local dataset.

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
    model = initialize_model(LogisticRegression, model_attributes)

    # Compute loss
    loss = log_loss(y, model.predict_proba(X))

    return {
        'loss': loss,
        'size': X.shape[0]
    }


@data(1)
def run_validation(
    df: pd.DataFrame, 
    parameters: List[np.ndarray], 
    classes: List[str], 
    predictors: List[str], 
    outcome: str
) -> Dict[str, Any]:
    """
    Validates logistic regression model on local dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Local data frame for validation.
    parameters : List[np.ndarray]
        Model parameters for validation.
    classes : List[str]
        List of class labels.
    predictors : List[str]
        Predictor variables for validation.
    outcome : str
        Outcome variable for validation.

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

    # Initialize LogisticRegression estimator
    model_attributes=dict(
            intercept_ = np.array(parameters[0]),
            coef_ = np.array(parameters[1]),
            classes_ = np.array(classes)
            )
    model = initialize_model(LogisticRegression, model_attributes)

    # Compute model accuracy
    score = model.score(X, y)

    # Compute confusion matrix
    confusion_matrix_ = confusion_matrix(
        y, model.predict(X), labels=model.classes_
    ).tolist()

    return {
        'score': score,
        'confusion_matrix': confusion_matrix_
    }