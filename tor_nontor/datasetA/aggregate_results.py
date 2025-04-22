import os
import pandas as pd
from glob import glob

# 所有结果所在路径
base_dir = "results"
run_folders = sorted(glob(os.path.join(base_dir, "run_*")))

# 需要聚合的文件名（你可按需删减或补充）
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

# 输出目录（可修改）
output_dir = "aggregated_results"
os.makedirs(output_dir, exist_ok=True)

# 开始聚合每类结果
for target_file in target_csv_files:
    all_dfs = []

    # 遍历每一个 run 文件夹
    for run in run_folders:
        csv_path = os.path.join(run, target_file)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["run"] = run  # 记录是哪一轮
            all_dfs.append(df)
        else:
            print(f"找不到：{csv_path}")

    if not all_dfs:
        print(f"未找到 {target_file}，跳过。")
        continue

    df_all = pd.concat(all_dfs, ignore_index=True)

    agg = (
        df_all.groupby(["perturbation", "level"])
        .agg(accuracy_mean=("accuracy", "mean"),
             accuracy_std=("accuracy", "std"))
        .reset_index()
    )

    # 输出聚合结果
    output_path = os.path.join(output_dir, target_file.replace(".csv", "_aggregated.csv"))
    agg.to_csv(output_path, index=False)
    print(f"已保存：{output_path}")
