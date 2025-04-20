import subprocess
import os
import shutil

# 所有要运行的脚本
scripts = [
    "batch_run_cnn_combined_perturbations.py",
    "batch_run_cnn_perturbations.py",
    "batch_run_knn_combined_perturbations.py",
    "batch_run_knn_perturbations.py",
    "batch_run_lstm_combined_perturbations.py",
    "batch_run_lstm_perturbations.py",
    "batch_run_rf_combined_perturbations.py",
    "batch_run_rf_perturbations.py",
    "batch_run_svm_combined_perturbations.py",
    "batch_run_svm_perturbations.py",
]

# 每个脚本运行后可能会产生的csv（你也可以根据实际情况更改）
expected_csv_files = {
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
}

# 执行次数
num_runs = 10

for i in range(1, num_runs + 1):
    print(f"\n🌀 开始第 {i} 轮测试...")
    run_dir = f"results/run_{i:02d}"
    os.makedirs(run_dir, exist_ok=True)

    # 运行每个脚本
    for script in scripts:
        print(f"🚀 运行脚本：{script}")
        subprocess.run(["python", script], check=True)

    # 移动所有生成的csv文件到对应文件夹
    for fname in expected_csv_files:
        if os.path.exists(fname):
            target_path = os.path.join(run_dir, fname)
            shutil.move(fname, target_path)
            print(f"✅ 移动 {fname} 到 {target_path}")
        else:
            print(f"⚠️ 警告：找不到结果文件 {fname}，可能该脚本没有生成？")

print("\n🎉 所有轮次运行完毕！结果已归档到 results/run_xx 文件夹中。")
