import subprocess
import os
import shutil

# all scripts
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

# generated csv file from each scripts
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

# run time
num_runs = 10

for i in range(1, num_runs + 1):
    print(f"\n Round {i} ...")
    run_dir = f"results/run_{i:02d}"
    os.makedirs(run_dir, exist_ok=True)

    for script in scripts:
        print(f" running script：{script}")
        subprocess.run(["python", script], check=True)

    # move generated csv
    for fname in expected_csv_files:
        if os.path.exists(fname):
            target_path = os.path.join(run_dir, fname)
            shutil.move(fname, target_path)
            print(f" Move {fname} to {target_path}")
        else:
            print(f"error: no found {fname}")

print("\n all done, results saving results/run_xx ")
