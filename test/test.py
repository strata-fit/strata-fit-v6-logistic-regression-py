from pathlib import Path

from sklearn.linear_model import LogisticRegression
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

from v6_logistic_regression_py.helper import initialize_model

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

    client = MockAlgorithmClient(datasets=datasets, module="v6_logistic_regression_py")
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


if __name__ == "__main__":
    main()
