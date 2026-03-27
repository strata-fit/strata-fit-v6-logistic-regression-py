from pathlib import Path
import tempfile

import pandas as pd
from sklearn.linear_model import LogisticRegression
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

from v6_federated_core import PartialFailureError
from v6_sklearn_linear_py.flower_bridge import _broadcast_fit_and_collect, _zeros_params
from v6_sklearn_linear_py.helper import initialize_model

STRATEGIES = [
    ("fedavg", {}),
    ("fedavgm", {}),
    ("fedadam", {}),
    ("fedyogi", {}),
    ("FaultTolerantFedAvg", {}),
]


def main() -> None:
    data_dir = Path("./data")
    datasets = [
        [{"database": data_dir / f"data_bucket{i}.csv", "db_type": "csv"}]
        for i in range(1, 5)
    ]

    client = MockAlgorithmClient(datasets=datasets, module="v6_sklearn_linear_py")
    org_ids = [org["id"] for org in client.organization.list()]

    partial_task = client.task.create(
        input_={
            "method": "logistic_regression_partial",
            "kwargs": {
                "model_attributes": {"coef_": [[0.0] * 6], "intercept_": [0.0]},
                "predictors": ["f0", "f1", "f2", "f3", "f4", "f5"],
                "outcome": "target",
                "n_local_iterations": 2,
                "penalty": "elasticnet",
                "solver": "saga",
                "l1_ratio": 0.5,
                "C": 0.1,
            },
        },
        organizations=[org_ids[0]],
    )
    partial_result = client.result.get(partial_task["id"])
    assert "model_attributes" in partial_result
    assert partial_result["size"] > 0

    for strategy_name, strategy_kwargs in STRATEGIES:
        master_task = client.task.create(
            input_={
                "master": True,
                "method": "master_flower",
                "kwargs": {
                    "org_ids": org_ids[:3],
                    "predictors": ["f0", "f1", "f2", "f3", "f4", "f5"],
                    "outcome": "target",
                    "classes": [0, 1],
                    "num_rounds": 2,
                    "n_local_epochs": 1,
                    "strategy_name": strategy_name,
                    "strategy_kwargs": strategy_kwargs,
                    "model_kwargs": {
                        "penalty": "elasticnet",
                        "solver": "saga",
                        "l1_ratio": 0.5,
                        "class_weight": "balanced",
                        "C": 0.1,
                    },
                },
            },
            organizations=[org_ids[0]],
        )
        master_result = client.result.get(master_task["id"])
        assert "model_attributes" in master_result, (strategy_name, master_result)
        assert len(master_result["history"]) == 2

        model = initialize_model(LogisticRegression, master_result["model_attributes"])
        assert model.coef_.shape[1] == 6

        validation_task = client.task.create(
            input_={
                "method": "run_validation",
                "kwargs": {
                    "parameters": master_result["model_attributes"],
                    "classes": [0, 1],
                    "predictors": ["f0", "f1", "f2", "f3", "f4", "f5"],
                    "outcome": "target",
                },
            },
            organizations=[org_ids[3]],
        )
        validation_result = client.result.get(validation_task["id"])
        assert 0.0 <= validation_result["score"] <= 1.0
        assert len(validation_result["confusion_matrix"]) == 2
        assert len(validation_result["confusion_matrix"][0]) == 2

    # Regression check: if a child partial fails and returns an error envelope,
    # master_flower should surface a structured partial failure (not KeyError).
    with tempfile.TemporaryDirectory(prefix="sklinear_failure_") as tmp_dir:
        fail_dir = Path(tmp_dir)
        node_1 = pd.read_csv(data_dir / "data_bucket1.csv").query("target == 0").iloc[:180]
        node_2 = pd.read_csv(data_dir / "data_bucket2.csv").query("target == 1").iloc[:180]
        node_3 = pd.read_csv(data_dir / "data_bucket3.csv").iloc[:220]
        node_4 = pd.read_csv(data_dir / "data_bucket4.csv").iloc[:200]

        node_1_path = fail_dir / "node_1.csv"
        node_2_path = fail_dir / "node_2.csv"
        node_3_path = fail_dir / "node_3.csv"
        node_4_path = fail_dir / "node_4.csv"
        node_1.to_csv(node_1_path, index=False)
        node_2.to_csv(node_2_path, index=False)
        node_3.to_csv(node_3_path, index=False)
        node_4.to_csv(node_4_path, index=False)

        failing_datasets = [
            [{"database": node_1_path, "db_type": "csv"}],
            [{"database": node_2_path, "db_type": "csv"}],
            [{"database": node_3_path, "db_type": "csv"}],
            [{"database": node_4_path, "db_type": "csv"}],
        ]
        failing_client = MockAlgorithmClient(datasets=failing_datasets, module="v6_sklearn_linear_py")
        failing_org_ids = [org["id"] for org in failing_client.organization.list()]

        try:
            failing_client.task.create(
                input_={
                    "master": True,
                    "method": "master_flower",
                    "kwargs": {
                        "org_ids": failing_org_ids[:3],
                        "predictors": ["f0", "f1", "f2", "f3", "f4", "f5"],
                        "outcome": "target",
                        "classes": [0, 1],
                        "num_rounds": 1,
                        "n_local_epochs": 1,
                        "strategy_name": "fedavg",
                        "strategy_kwargs": {},
                        "model_kwargs": {
                            "penalty": "l2",
                            "solver": "lbfgs",
                        },
                    },
                },
                organizations=[failing_org_ids[0]],
            )
            raise AssertionError("Expected one-class local node failure to raise")
        except ValueError as exc:
            text = str(exc)
            assert "at least 2 classes" in text, text
            assert "'model_attributes'" not in text, text

    class _FakeTaskAPI:
        def create(self, **kwargs):
            return {"id": 123}

    class _FakeClient:
        def __init__(self, results):
            self.task = _FakeTaskAPI()
            self._results = results

        def wait_for_results(self, task_id: int, interval: int = 1):
            return self._results

    params = _zeros_params(n_classes=2, n_features=6)
    fake_client_empty = _FakeClient(results=[])
    try:
        _broadcast_fit_and_collect(
            client=fake_client_empty,
            org_ids=[1, 2, 3],
            params=params,
            predictors=["f0", "f1", "f2", "f3", "f4", "f5"],
            outcome="target",
            database_label="default",
            n_local_epochs=1,
            model_kwargs={},
        )
        raise AssertionError("Expected empty result set to raise PartialFailureError")
    except PartialFailureError as exc:
        assert "No node fit results" in str(exc)

    fake_client_incomplete = _FakeClient(
        results=[
            {
                "model_attributes": {"coef_": [[0.0] * 6], "intercept_": [0.0]},
                "size": 10,
            }
        ]
    )
    try:
        _broadcast_fit_and_collect(
            client=fake_client_incomplete,
            org_ids=[1, 2, 3],
            params=params,
            predictors=["f0", "f1", "f2", "f3", "f4", "f5"],
            outcome="target",
            database_label="default",
            n_local_epochs=1,
            model_kwargs={},
        )
        raise AssertionError("Expected incomplete result set to raise PartialFailureError")
    except PartialFailureError as exc:
        assert "Incomplete node fit results" in str(exc)


if __name__ == "__main__":
    main()
