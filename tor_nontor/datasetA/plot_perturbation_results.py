import os
import pandas as pd
import matplotlib.pyplot as plt

# 模型名列表
model_names = ["cnn", "knn", "lstm", "rf", "svm"]
# 聚合结果所在文件夹
base_dir = "aggregated_results"

# 存储每个模型的数据
model_data = {}

# 读取每个模型的聚合CSV（单独扰动 + 联合扰动）
for model in model_names:
    dfs = []

    single_path = os.path.join(base_dir, f"{model}_perturbation_results_aggregated.csv")
    if os.path.exists(single_path):
        df_single = pd.read_csv(single_path)
        dfs.append(df_single)

    combined_path = os.path.join(base_dir, f"{model}_perturbation_all_combined_aggregated.csv")
    if os.path.exists(combined_path):
        df_combined = pd.read_csv(combined_path)
        df_combined["perturbation"] = "combined_all"
        dfs.append(df_combined)

    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        model_data[model] = df_all

# ========== 第1~5张图：每个模型一张，5条线 ========== #
for model, df in model_data.items():
    plt.figure(figsize=(8, 5))
    for pert in sorted(df["perturbation"].unique()):
        sub_df = df[df["perturbation"] == pert]
        plt.plot(sub_df["level"], sub_df["accuracy_mean"], marker='o', label=pert)
    plt.title(f"{model.upper()} Accuracy vs Perturbation Level")
    plt.xlabel("Perturbation Level")
    plt.ylabel("Accuracy (Mean)")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model}_accuracy_vs_perturbation_lines.png")
    plt.close()

# ========== 第6张图：所有模型在混合扰动下的表现 ========== #
plt.figure(figsize=(8, 5))
for model, df in model_data.items():
    comb_df = df[df["perturbation"] == "combined_all"]
    if not comb_df.empty:
        plt.plot(comb_df["level"], comb_df["accuracy_mean"], marker='o', label=model)
plt.title("Combined Perturbation - Accuracy Across Models")
plt.xlabel("Perturbation Level")
plt.ylabel("Accuracy (Mean)")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("all_models_combined_accuracy.png")
plt.close()
