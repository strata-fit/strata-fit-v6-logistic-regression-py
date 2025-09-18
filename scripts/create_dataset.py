from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from pathlib import Path
import tomllib 

CONFIG_FILE = "scripts/create_dataset.toml"

def main():
    # Load config
    with open(CONFIG_FILE, "rb") as f:
        cfg = tomllib.load(f)

    g = cfg["global"]
    output_folder = Path(g["output_folder"])
    output_folder.mkdir(parents=True, exist_ok=True)

    # Generate a global dataset
    X, y = make_classification(
        n_samples=g["n_samples"], 
        n_features=g["n_features"], 
        n_informative=g["n_informative"], 
        n_redundant=g["n_redundant"],
        n_classes=g["n_classes"], 
        weights=g["weights"], 
        random_state=g["random_state"],
    )

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y

    # Split into 4 buckets with intentional skew
    for i in range(1, 5):
        bucket = df.sample(frac=0.25, random_state=g["random_state"] + i)

        bucket_cfg = cfg["buckets"].get(f"bucket{i}", {})
        if "skew_class" in bucket_cfg:
            skew_class = bucket_cfg["skew_class"]
            other_frac = bucket_cfg.get("other_frac", 0.5)
            # Keep all from skew_class, only a fraction from the other
            bucket_major = bucket[bucket["target"] == skew_class]
            bucket_minor = bucket[bucket["target"] != skew_class].sample(
                frac=other_frac, random_state=g["random_state"]+i
            )
            bucket = pd.concat([bucket_major, bucket_minor], ignore_index=True)
        if "shift_feature" in bucket_cfg:
            feat = bucket_cfg["shift_feature"]
            shift = bucket_cfg.get("shift_value", 0.0)
            bucket[feat] = bucket[feat] + shift

        bucket.to_csv(output_folder / f"data_bucket{i}.csv", index=False)

    print(f"Generated buckets in {output_folder}")

if __name__ == "__main__":
    main()
