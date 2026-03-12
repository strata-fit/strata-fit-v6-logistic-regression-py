# Vantage6 sklearn linear package

This package is designed for the [vantage6](https://vantage6.ai/) architecture.

Research-only notice: this code may still show unexpected behavior and should be
treated as research/prototyping software for now.

## Supported methods

- `master_flower`
- `logistic_regression_partial`
- `compute_loss_partial`
- `run_validation`

## Sending tasks with GHCR images

Nodes do not need a local clone of this repository. They only need network
access and permissions to pull the image you submit in `task.create(image=...)`.

Use a GHCR image reference, for example:

- `ghcr.io/strata-fit/v6-sklearn-linear-py:latest`

If the GHCR package is private, configure Docker credentials on each node host
so node containers can pull from GHCR.

### Example: binary classification task

```python
import time
from vantage6.client import Client

client = Client("http://127.0.0.1", 5070, "/api")
client.authenticate("gamma-user", "gamma-password")
client.setup_encryption(None)

collab_id = 1
master_org_id = 3
org_ids = [1, 2, 3, 4, 5]

input_ = {
    "master": True,
    "method": "master_flower",
    "kwargs": {
        "org_ids": org_ids,
        "predictors": ["f0", "f1", "f2", "f3", "f4", "f5"],
        "outcome": "target",
        "classes": [0, 1],
        "database_label": "default",
        "num_rounds": 2,
        "n_local_epochs": 1,
        "strategy_name": "fedavg",
        "strategy_kwargs": {},
        "model_kwargs": {
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": "balanced",
            "C": 1.0,
            "max_iter": 200,
        },
    },
}

task = client.task.create(
    collaboration=collab_id,
    organizations=[master_org_id],
    name="ghcr-binary-l2",
    image="ghcr.io/strata-fit/v6-sklearn-linear-py:latest",
    description="binary logistic l2",
    input_=input_,
    databases=[{"label": "default"}],
)

while True:
    status = client.task.get(task["id"])["status"]
    if status in {"completed", "failed", "crashed", "cancelled", "non-existing Docker image"}:
        break
    time.sleep(2)

print("task status:", status)
print("result rows:", client.result.from_task(task["id"])["data"])
```

### Example: continuous outcome task

For continuous/regression-style runs, pass `classes=None`:

```python
input_["kwargs"].update(
    {
        "predictors": ["f1", "f2", "f3", "f4", "f5"],
        "outcome": "f0",
        "classes": None,
        "model_kwargs": {
            "model_class": "ElasticNet",
            "alpha": 0.1,
            "l1_ratio": 0.5,
            "max_iter": 300,
        },
    }
)
```

## Testing locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python test/test.py
```

## Manual Infra Smoke in GitHub Actions

Workflow:

- `.github/workflows/manual_infra_smoke.yml`

It runs end-to-end infra-backed smoke tests using configurable node counts and
vantage6 version defaults (currently `4.13.3`).
