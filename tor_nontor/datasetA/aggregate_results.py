import os
import pandas as pd
from glob import glob

base_dir = "results"
run_folders = sorted(glob(os.path.join(base_dir, "run_*")))

target_csv_files = [
    "cnn_perturbation_results.csv",
    "cnn_perturbation_all_combined.csv",
    "knn_perturbation_results.csv",
    "knn_perturbation_all_combined.csv",
    "lstm_perturbation_results.csv",
    "lstm_perturbation_all_combined.csv",
    "rf_perturbation_results.csv",
    "rf_perturbation_all_combined.csv",
    "svm_perturbation_results.csv",
    "svm_perturbation_all_combined.csv",
]

output_dir = "aggregated_results"
os.makedirs(output_dir, exist_ok=True)

for target_file in target_csv_files:
    all_dfs = []

    for run in run_folders:
        csv_path = os.path.join(run, target_file)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["run"] = run  
            all_dfs.append(df)
        else:
            print(f"error in : {csv_path}")

    if not all_dfs:
        print(f"no found {target_file}")
        continue

    df_all = pd.concat(all_dfs, ignore_index=True)

    agg = (
        df_all.groupby(["perturbation", "level"])
        .agg(accuracy_mean=("accuracy", "mean"),
             accuracy_std=("accuracy", "std"))
        .reset_index()
    )

    # aggregate all result
    output_path = os.path.join(output_dir, target_file.replace(".csv", "_aggregated.csv"))
    agg.to_csv(output_path, index=False)
    print(f"saved：{output_path}")
