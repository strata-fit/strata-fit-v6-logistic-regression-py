from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client
from v6_federated_core import MethodContext, dispatch_registered_method, to_v6_result

from flwr.common import (
    Parameters,
    FitRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import (
    FedAvg,
    FedAdam,
    FedYogi,
    FedAvgM,
    FaultTolerantFedAvg,
    # "Bulyan",
    # "DPFedAvgAdaptive",
    # "DPFedAvgFixed",
    # "DifferentialPrivacyClientSideAdaptiveClipping",
    # "DifferentialPrivacyClientSideFixedClipping",
    # "DifferentialPrivacyServerSideAdaptiveClipping",
    # "DifferentialPrivacyServerSideFixedClipping",
    # "FaultTolerantFedAvg",
    # "FedAdagrad",
    # "FedAdam",
    # "FedAvg",
    # "FedAvgAndroid",
    # "FedAvgM",
    # "FedMedian",
    # "FedOpt",
    # "FedProx",
    # "FedTrimmedAvg",
    # "FedXgbBagging",
    # "FedXgbCyclic",
    # "FedXgbNnAvg",
    # "FedYogi",
    # "Krum",
    # "QFedAvg",
    # "Strategy",
)

# ---- Strategy registry (server-side only; no client-code changes required) ----
_STRATEGIES = {
    "fedavg": lambda **cfg: FedAvg(**cfg),
    "fedadam": lambda **cfg: FedAdam(**cfg),
    "fedyogi": lambda **cfg: FedYogi(**cfg),
    "fedavgm": lambda **cfg: FedAvgM(**cfg),
    "FaultTolerantFedAvg".lower(): lambda **cfg: FaultTolerantFedAvg(**cfg),
    # Note: FedProx requires client-side proximal term in the loss;
    # with scikit-learn LR we cannot inject it, so it won't behave as intended.
}

# Strategies which need a Flower Parameters at construction
_NEEDS_INIT = {"fedadam", "fedadagrad", "fedyogi", "fedavgm", "FaultTolerantFedAvg".lower()}

def _make_initial_parameters(n_features: int, n_classes: int) -> Parameters:
    rows = 1 if n_classes <= 2 else n_classes
    coef = np.zeros((rows, n_features), dtype=np.float64)
    inter = np.zeros((rows,), dtype=np.float64)
    return ndarrays_to_parameters([coef, inter])

def _strategy_from_name(name: str, **cfg):
    """Create a Flower strategy, optionally injecting initial_parameters.

    Internal-only keys:
      - n_features, n_classes : used to build zero init if needed, then popped.
    """
    key = name.strip().lower()
    if key not in _STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}' (available: {list(_STRATEGIES.keys())})")

    # --- pop internal hints so they are NOT passed to the strategy constructor ---
    n_features = cfg.pop("n_features", None)
    n_classes  = cfg.pop("n_classes", None)

    # --- ensure initial_parameters is a real Flower Parameters for FedOpt-family ---
    if key in {"fedadam", "fedadagrad", "fedyogi", "fedavgm"}:
        ip = cfg.get("initial_parameters", None)
        if ip is None:
            if n_features is None or n_classes is None:
                raise ValueError(
                    "Strategy requires initial_parameters; supply it explicitly "
                    "or pass n_features and n_classes so the bridge can build zeros."
                )
            cfg["initial_parameters"] = _make_initial_parameters(n_features, n_classes)
        else:
            # make robust: accept lists/np arrays or our JSON-style {"arrays":[...]}
            try:
                _ = ip.tensors  # already a Flower Parameters
            except AttributeError:
                if isinstance(ip, dict) and "arrays" in ip:
                    nds = [np.array(a) for a in ip["arrays"]]
                else:
                    nds = [np.array(a) for a in ip]
                cfg["initial_parameters"] = ndarrays_to_parameters(nds)

    return _STRATEGIES[key](**cfg)

def _model_attrs_to_ndarrays(ma: Dict[str, Any]) -> List[np.ndarray]:
    """Order: [coef_, intercept_] -> list of ndarrays."""
    coef = np.array(ma["coef_"])
    inter = np.array(ma["intercept_"])
    return [coef, inter]

def _ndarrays_to_model_attrs(nds: List[np.ndarray], classes: np.ndarray) -> Dict[str, Any]:
    return {"coef_": nds[0], "intercept_": nds[1], "classes_": classes}

def _params_to_payload(p: Parameters) -> Dict[str, Any]:
    "Flower Parameters -> JSON payload"
    nds = parameters_to_ndarrays(p)
    return {"arrays": [a.tolist() for a in nds]}

def _payload_to_params(payload: Dict[str, Any]) -> Parameters:
    "JSON payload -> Flower Parameters"
    nds = [np.array(a) for a in payload["arrays"]]
    return ndarrays_to_parameters(nds)

def _zeros_params(n_classes: int, n_features: int) -> Parameters:
    rows = 1 if n_classes <= 2 else n_classes
    coef = np.zeros((rows, n_features))
    inter = np.zeros((rows,))
    return ndarrays_to_parameters([coef, inter])

def _mk_fitres(updated_params: Parameters, num_examples: int, metrics: Dict[str, float]) -> FitRes:
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=updated_params,
        num_examples=num_examples,
        metrics=metrics,
    )


def _broadcast_fit_and_collect(
    client: AlgorithmClient,
    org_ids: List[int],
    params: Parameters,
    predictors: List[str],
    outcome: str,
    database_label: str,
    n_local_epochs: int,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Tuple[None, FitRes]]:
    """
    Calls your existing partial once per node:
      method='logistic_regression_partial'
      kwargs={'model_attributes': ..., 'predictors': ..., 'outcome': ...}
    Returns Flower-like FitRes entries for strategy.aggregate_fit.
    """
    # Convert Parameters -> model_attributes payload the partial expects
    nds = parameters_to_ndarrays(params)
    # classes_ are not embedded in Flower Parameters; pass None here.
    # Your partial builds classes_ from data or can accept it via kwargs if desired.
    model_attributes = {"coef_": nds[0].tolist(), "intercept_": nds[1].tolist()}

    base_kwargs = {
        "model_attributes": model_attributes,
        "predictors": predictors,
        "outcome": outcome,
        "n_local_iterations": n_local_epochs,
        **(model_kwargs or {}),
    }

    input_ = {"method": "logistic_regression_partial", "kwargs": base_kwargs}

    task_create_kwargs = {
        "input_": input_,
        "organizations": org_ids,
        "databases": [{"label": database_label}],
    }
    try:
        task = client.task.create(**task_create_kwargs)
    except TypeError:
        # MockAlgorithmClient does not accept `databases`; retry without it.
        task_create_kwargs.pop("databases", None)
        task = client.task.create(**task_create_kwargs)
    results = client.wait_for_results(task_id=task["id"], interval=1)

    fit_results: List[Tuple[None, FitRes]] = []
    for res in results:
        if isinstance(res, dict) and "ok" in res:
            error_messages = ", ".join(
                error.get("message", "unknown error") for error in res.get("errors", [])
            )
            raise RuntimeError(
                "Node fit task returned a failure envelope"
                + (f": {error_messages}" if error_messages else "")
            )
        # res: {'model_attributes': {...}, 'size': int}
        ma = res["model_attributes"]
        nds_upd = _model_attrs_to_ndarrays(ma)
        updated_params = ndarrays_to_parameters(nds_upd)
        nexp = int(res["size"])
        fit_results.append((None, _mk_fitres(updated_params, nexp, metrics={})))
    return fit_results


# -----------------------------
# Public master entrypoint (V6)
# -----------------------------
@algorithm_client
def master_flower(
    client: AlgorithmClient,
    *,
    org_ids: List[int],
    predictors: List[str],
    outcome: str,
    classes: List[Any],
    database_label: str = "default",
    num_rounds: int = 5,
    n_local_epochs: int = 1,
    strategy_name: str = "fedavg",
    # Optional strategy kwargs, e.g., server_learning_rate for FedAdam/FedYogi
    strategy_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from v6_logistic_regression_py.methods import METHOD_REGISTRY

    envelope = dispatch_registered_method(
        METHOD_REGISTRY,
        "master_flower",
        {
            "org_ids": org_ids,
            "predictors": predictors,
            "outcome": outcome,
            "classes": classes,
            "database_label": database_label,
            "num_rounds": num_rounds,
            "n_local_epochs": n_local_epochs,
            "strategy_name": strategy_name,
            "strategy_kwargs": strategy_kwargs or {},
            "model_kwargs": model_kwargs or {},
        },
        context=MethodContext(method="master_flower", meta={"client": client}),
    )
    return to_v6_result(envelope)


def _master_flower_core(
    client: AlgorithmClient,
    *,
    org_ids: List[int],
    predictors: List[str],
    outcome: str,
    classes: List[Any],
    database_label: str = "default",
    num_rounds: int = 5,
    n_local_epochs: int = 1,
    strategy_name: str = "fedavg",
    strategy_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Vantage6 master that runs Flower's server-side strategy loop
    while delegating local fits to V6 one-shot partials.
    """
    info(f"Starting Flower(master) over V6 RPC | strategy={strategy_name}")
    strategy_kwargs = strategy_kwargs or {}
    model_kwargs = model_kwargs or {}

    n_features = len(predictors)
    n_classes = len(classes)
    classes_arr = np.array(classes)

    # Initial global params (zeros)
    global_params = _zeros_params(n_classes, n_features)

    # Build strategy
    strat = _strategy_from_name(
        strategy_name,
        n_features=n_features,
        n_classes=n_classes,
        **strategy_kwargs,
    )

    history = []
    for rnd in range(1, num_rounds + 1):
        info(f"[master_flower] Round {rnd} - dispatch fit partials")
        # Dispatch one-shot local fits
        fit_results = _broadcast_fit_and_collect(
            client=client,
            org_ids=org_ids,
            params=global_params,
            predictors=predictors,
            outcome=outcome,
            database_label=database_label,
            n_local_epochs=n_local_epochs,
            # Pass-through extras
            model_kwargs=model_kwargs
        )

        # Aggregate with Flower strategy
        agg = strat.aggregate_fit(rnd, fit_results, failures=[])
        if agg is None:
            raise RuntimeError("Aggregation returned None")
        global_params, _ = agg

        # (Optional) Evaluate similarly via a compute_loss/eval partial
        # and strat.aggregate_evaluate(...)
        

        # Keep minimal trace
        tot_examples = int(sum(fr.num_examples for _, fr in fit_results))
        history.append({"round": rnd, "num_examples": tot_examples})

    # Return final model as your V6-style attribute dict
    final_nds = parameters_to_ndarrays(global_params)
    final_attrs = _ndarrays_to_model_attrs(final_nds, classes_arr)
    info("[master_flower] Finished")
    return {
        "model_attributes": {
            "coef_": final_attrs["coef_"].tolist(),
            "intercept_": final_attrs["intercept_"].tolist(),
            "classes_": final_attrs["classes_"].tolist(),
        },
        "history": history,
    }
