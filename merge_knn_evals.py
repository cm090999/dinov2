import pandas as pd
import os

if __name__ == "__main__":
    csv_paths = [
        "/home/ubuntu/localstorage/dinov2/results/knn_eval/loradinoret_20250715_090459/eval/eval_results_iteration_0_step_0_aptos.csv",
        "/home/ubuntu/localstorage/dinov2/results/knn_eval/loradinoret_20250715_090459/eval/eval_results_iteration_0_step_0_imagenet.csv",
        "/home/ubuntu/localstorage/dinov2/results/knn_eval/dinov2_vitb_20250715_085659/eval/eval_results_iteration_0_step_0_aptos.csv",
        "/home/ubuntu/localstorage/dinov2/results/knn_eval/dinov2_vitb_20250715_085659/eval/eval_results_iteration_0_step_0_imagenet.csv",
        "/home/ubuntu/localstorage/dinov2/results/knn_eval/dinoret_20250715_090054/eval/eval_results_iteration_0_step_0_aptos.csv",
        "/home/ubuntu/localstorage/dinov2/results/knn_eval/dinoret_20250715_090054/eval/eval_results_iteration_0_step_0_imagenet.csv",
        "/home/ubuntu/localstorage/dinov2/results/knn_eval/bedinoret_20250715_090630/eval/eval_results_iteration_0_step_0_aptos.csv",
        "/home/ubuntu/localstorage/dinov2/results/knn_eval/bedinoret_20250715_090630/eval/eval_results_iteration_0_step_0_imagenet.csv"
    ]
    output_csv_path = "/home/ubuntu/localstorage/dinov2/results/knn_eval/merged_knn_eval_results.csv"
    extract_dataset_names = lambda x: x.split("/")[-1].split("_")[-1].split(".")[0]
    extract_model_name = lambda x: x.split("/")[-3].split("_")[0]
    dataset_names = [extract_dataset_names(path) for path in csv_paths]
    model_names = [extract_model_name(path) for path in csv_paths]
    dfs = [pd.read_csv(path) for path in csv_paths]
    
    # Add model names to each dataframe
    for i, df in enumerate(dfs):
        df['model'] = model_names[i]
    
    merged_df = pd.concat(dfs, keys=dataset_names, names=["dataset", "index"])
    merged_df = merged_df.reset_index(level=0)
    merged_df.to_csv(output_csv_path, index=False)
    