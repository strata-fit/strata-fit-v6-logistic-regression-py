#!/usr/bin/env python3
"""Submit smoke tasks against local vantage6 infra and validate completion."""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from vantage6.client import Client

TERMINAL_STATUSES = {
    "completed",
    "crashed",
    "failed",
    "cancelled",
    "non-existing Docker image",
}


@dataclass
class TaskSpec:
    name: str
    org_names: list[str]
    model_kwargs: dict[str, Any]


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def decode_result(value: Any) -> Any:
    if value is None or isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {"raw": str(value)}

    try:
        return json.loads(base64.b64decode(value).decode("utf-8"))
    except Exception:
        pass

    try:
        return json.loads(value)
    except Exception:
        return {"raw": value}


def wait_for_terminal(client: Client, task_id: int, timeout_s: int) -> str:
    status = None
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        current = client.task.get(task_id).get("status")
        if current != status:
            print(f"task {task_id} status: {current}")
            status = current
        if current in TERMINAL_STATUSES:
            return current
        time.sleep(2)
    raise TimeoutError(f"Task {task_id} did not finish before timeout")


def get_child_tasks(client: Client, parent_task_id: int) -> list[dict[str, Any]]:
    page = 1
    all_children: list[dict[str, Any]] = []
    while True:
        payload = client.task.list(parent=parent_task_id, page=page, per_page=50)
        chunk = payload.get("data", [])
        all_children.extend(chunk)
        if len(chunk) < 50:
            break
        page += 1
    return all_children


def validate_task(
    client: Client,
    task_id: int,
    expected_org_count: int,
    expected_rounds: int,
) -> None:
    results = client.result.from_task(task_id).get("data", [])
    if not results:
        raise RuntimeError(f"Task {task_id} has no results")

    decoded = decode_result(results[0].get("result"))
    if isinstance(decoded, dict) and decoded.get("ok") is False:
        raise RuntimeError(f"Task {task_id} returned error envelope: {decoded}")

    if not isinstance(decoded, dict):
        raise RuntimeError(f"Task {task_id} result is not a dict: {decoded}")

    required_keys = {"model_attributes", "history"}
    missing = required_keys.difference(decoded.keys())
    if missing:
        raise RuntimeError(f"Task {task_id} missing keys: {sorted(missing)}")

    children = get_child_tasks(client, task_id)
    if not children:
        raise RuntimeError(f"Task {task_id} has no child tasks")

    run_statuses: list[str] = []
    run_count = 0
    for child in children:
        child_runs = client.run.from_task(child["id"]).get("data", [])
        run_count += len(child_runs)
        run_statuses.extend([run.get("status") for run in child_runs])

    expected_runs = expected_org_count * expected_rounds
    if run_count < expected_runs:
        raise RuntimeError(
            f"Task {task_id} has too few runs: got {run_count}, expected at least {expected_runs}"
        )

    failed = [status for status in run_statuses if status != "completed"]
    if failed:
        raise RuntimeError(f"Task {task_id} has non-completed child runs: {failed}")

    print(
        f"task {task_id} validated: keys={sorted(decoded.keys())}, "
        f"child_runs={run_count}, child_tasks={len(children)}"
    )


def main() -> None:
    host = os.getenv("V6_SERVER_HOST", "http://localhost")
    port = env_int("V6_SERVER_PORT", 5070)
    api_path = os.getenv("V6_API_PATH", "/api")
    collaboration_name = os.getenv("V6_COLLABORATION_NAME", "linear-ci")
    image = os.environ["V6_ALGO_IMAGE"]

    node_count = env_int("V6_NODE_COUNT", 5)
    elasticnet_node_count = env_int("V6_ELASTICNET_NODE_COUNT", 4)
    num_rounds = env_int("V6_NUM_ROUNDS", 2)
    n_local_epochs = env_int("V6_N_LOCAL_EPOCHS", 1)
    timeout_s = env_int("V6_TASK_TIMEOUT_S", 900)

    run_l2 = env_bool("V6_RUN_L2", True)
    run_elasticnet = env_bool("V6_RUN_ELASTICNET", True)

    ordered_names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    selected = ordered_names[:node_count]
    selected_elastic = ordered_names[:elasticnet_node_count]

    if elasticnet_node_count > node_count:
        raise ValueError("V6_ELASTICNET_NODE_COUNT cannot exceed V6_NODE_COUNT")

    client = Client(host, port, api_path)

    master = "gamma" if "gamma" in selected else selected[0]
    client.authenticate(f"{master}-user", f"{master}-password")
    client.setup_encryption(None)

    collab = next(
        c for c in client.collaboration.list()["data"] if c["name"] == collaboration_name
    )
    collab_id = collab["id"]

    org_map = {o["name"]: o["id"] for o in client.organization.list()["data"]}

    specs: list[TaskSpec] = []
    if run_l2:
        specs.append(
            TaskSpec(
                name=f"ci-logistic-l2-{node_count}nodes",
                org_names=selected,
                model_kwargs={
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "class_weight": "balanced",
                    "C": 1.0,
                    "max_iter": 200,
                },
            )
        )

    if run_elasticnet:
        specs.append(
            TaskSpec(
                name=f"ci-logistic-elasticnet-{elasticnet_node_count}nodes",
                org_names=selected_elastic,
                model_kwargs={
                    "penalty": "elasticnet",
                    "solver": "saga",
                    "l1_ratio": 0.5,
                    "class_weight": "balanced",
                    "C": 0.1,
                    "max_iter": 200,
                },
            )
        )

    if not specs:
        raise ValueError("At least one of V6_RUN_L2 or V6_RUN_ELASTICNET must be true")

    for spec in specs:
        org_ids = [org_map[name] for name in spec.org_names]
        master_org = org_map[master]

        input_ = {
            "master": True,
            "method": "master_flower",
            "kwargs": {
                "org_ids": org_ids,
                "predictors": ["f0", "f1", "f2", "f3", "f4", "f5"],
                "outcome": "target",
                "classes": [0, 1],
                "database_label": "default",
                "num_rounds": num_rounds,
                "n_local_epochs": n_local_epochs,
                "strategy_name": "fedavg",
                "strategy_kwargs": {},
                "model_kwargs": spec.model_kwargs,
            },
        }

        task = client.task.create(
            collaboration=collab_id,
            organizations=[master_org],
            name=spec.name,
            image=image,
            description=spec.name,
            input_=input_,
            databases=[{"label": "default"}],
        )

        task_id = task["id"]
        print(
            f"created task {task_id} name={spec.name} "
            f"master={master} organizations={spec.org_names}"
        )

        status = wait_for_terminal(client, task_id, timeout_s)
        if status != "completed":
            raise RuntimeError(f"Task {task_id} finished with status '{status}'")

        validate_task(
            client=client,
            task_id=task_id,
            expected_org_count=len(org_ids),
            expected_rounds=num_rounds,
        )

    print("infra smoke algorithm tasks completed")


if __name__ == "__main__":
    main()
