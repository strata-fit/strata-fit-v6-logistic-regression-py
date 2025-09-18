from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression, ElasticNet
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
from v6_logistic_regression_py import master_flower
from v6_logistic_regression_py.helper import initialize_model

data_dir = Path('./data')
ds1 = {"database": data_dir/"data_bucket1.csv", "db_type": "csv"}
ds2 = {"database": data_dir/"data_bucket2.csv", "db_type": "csv"}
ds3 = {"database": data_dir/"data_bucket3.csv", "db_type": "csv"}
ds4 = {"database": data_dir/"data_bucket4.csv", "db_type": "csv"}

datasets = [[ds1], [ds2], [ds3], [ds4]]


client = MockAlgorithmClient(datasets=datasets, module='v6_logistic_regression_py')
orgs = client.organization.list()
org_ids = [o["id"] for o in orgs]

model_kwargs = {
    'penalty': 'elasticnet',
    'solver': 'saga',
    'l1_ratio': 0.5,
    'class_weight': 'balanced',
    'C': 0.1,
}
num_iterations = 50

master_task = client.task.create(
    input_={
        'master': True,
        'method': 'master_flower',
        'kwargs': {
            'org_ids': org_ids[:3],                      # all orgs participate
            'predictors': ['f0', 'f1', 'f2', 'f3', 'f4', 'f5'],
            'outcome': 'target',
            'classes': [0, 1],
            'num_rounds': num_iterations,                     # Flower rounds
            'n_local_epochs': 1,                 # scikit-learn local epochs per round
            'strategy_name': 'fedavg',                # try 'fedadam' or 'fedyogi' too
            'strategy_kwargs': {
                # 'eta': 0.1,          # server lr (FedOpt family)
                # optional extras, e.g.:
                # 'beta_1': 0.9, 'beta_2': 0.99, 'tau': 1e-9, 'eta_l': 0.1,
                # For FedAvgM: 'server_learning_rate': 1.0, 'server_momentum': 0.9
            },
            'model_kwargs': model_kwargs,
        }
    },
    organizations=[org_ids[0]]  # run the master at a single org
)
results = client.result.get(master_task.get('id'))

# Rebuild model from returned attributes
model = initialize_model(LogisticRegression, results['model_attributes'])
history = results.get('history', [])
print("Final coef/intercept:", model.coef_.shape, model.intercept_.shape)
print(f"Rounds completed: {len(history)}")

print([model.intercept_.tolist(), model.coef_.tolist()])

# --- Validate using your existing validation partial ---
val_task = client.task.create(
    input_={
        'master': False,
        'method': 'run_validation',
        'kwargs': {

            'parameters': [model.intercept_.tolist(), model.coef_.tolist()],
            'classes': [0, 1],
            'predictors': ['f0', 'f1', 'f2', 'f3', 'f4', 'f5'],
            'outcome': 'target',
        }
    },
    organizations=[org_ids[3]]
)
val_results = client.result.get(val_task.get('id'))
accuracy = val_results['score']
cm = val_results['confusion_matrix']
print(f'Federated model accuracy: {accuracy}')
print(f'Federated model confusion matrix: {cm}')


# -------------------
# Centralized baseline
# -------------------
df_all = pd.concat([pd.read_csv(ds[0]["database"]) for ds in datasets[:3]], ignore_index=True)
X_train = df_all[["f0", "f1", "f2", "f3", "f4", "f5"]].values
y_train = df_all["target"].values

df_all = pd.concat([pd.read_csv(ds[0]["database"]) for ds in datasets[3:]], ignore_index=True)
X_test = df_all[["f0", "f1", "f2", "f3", "f4", "f5"]].values
y_test = df_all["target"].values


central_model = LogisticRegression(
    max_iter=num_iterations,  # allow convergence
    **model_kwargs,
)
central_model.fit(X_train, y_train)
central_accuracy = central_model.score(X_test, y_test)
print(f"Central model accuracy: {central_accuracy:.4f}")
print(f"Central model confusion matrix: {confusion_matrix(y_test, central_model.predict(X_test)).tolist()}")