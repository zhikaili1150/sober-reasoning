import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer


def analyze_parquet_folder(
    folder_path: str,
    task: str,
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    save_csv: bool = False,
):
    """
    递归遍历指定文件夹下的所有 .parquet 文件，
    计算平均 token length 与 extractive_match。

    参数:
        folder_path (str): 要分析的文件夹路径
        model_name (str): 用于分词的模型名称
        save_csv (bool): 是否保存详细结果 CSV (默认 False)

    返回:
        dict: {"avg_token_length": float, "avg_extractive_match": float}
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # === 收集所有 parquet 文件 ===
    parquet_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(".parquet") and task in f:
                parquet_files.append(os.path.join(root, f))

    if not parquet_files:
        print(f"⚠️ No parquet files found under {folder_path}")
        return {"avg_token_length": np.nan, "avg_extractive_match": np.nan}

    print(f"🔍 Found {len(parquet_files)} parquet files under {folder_path}")

    records = []

    for parquet_path in parquet_files:
        try:
            details = load_dataset("parquet", data_files=parquet_path, split="train")
            token_lengths, extractive_matches = [], []

            for item in details:
                preds = item.get("predictions", [])
                if preds and isinstance(preds[0], str):
                    tokens = tokenizer.encode(preds[0])
                    token_lengths.append(len(tokens))

                metrics = item.get("metrics", {})
                if isinstance(metrics, dict) and "extractive_match" in metrics:
                    extractive_matches.append(metrics["extractive_match"])

            if token_lengths or extractive_matches:
                avg_len = np.mean(token_lengths) if token_lengths else np.nan
                avg_em = np.mean(extractive_matches) if extractive_matches else np.nan
                records.append({
                    "file": os.path.basename(parquet_path),
                    "avg_token_length": avg_len,
                    "avg_extractive_match": avg_em
                })
                print(f"📄 {os.path.basename(parquet_path)} → tokens={avg_len:.1f}, extractive_match={avg_em:.3f}")

        except Exception as e:
            print(f"❌ Failed to process {parquet_path}: {e}")

    if not records:
        print("⚠️ No valid data found.")
        return {"avg_token_length": np.nan, "avg_extractive_match": np.nan}

    df = pd.DataFrame(records)
    avg_len = df["avg_token_length"].mean()
    avg_em = df["avg_extractive_match"].mean()

    print(f"\n✅ Folder summary → tokens={avg_len:.1f}, extractive_match={avg_em:.3f}")

    if save_csv:
        output_csv = os.path.join(folder_path, "aggregated_metrics.csv")
        df.to_csv(output_csv, index=False)
        print(f"💾 Detailed results saved to: {output_csv}")

    return {"avg_token_length": avg_len, "avg_extractive_match": avg_em}

import os
import pandas as pd

if __name__ == "__main__":
    # === 参数设置 ===
    parent_dir = "/local/scratch/zli2255/workspace/sober-reasoning/result/merge_method/dare_ties/accfmt/details"
    output_csv = os.path.join(parent_dir, "summary.csv")
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    task_list = ["aime24", "aime25", "amc23", "math_500", "minerva", "olympiadbench"]

    # 每个 folder_name 汇总成一行
    folder_dict = {}

    for folder_name in sorted(os.listdir(parent_dir)):
        folder_path = os.path.join(parent_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        print(f"\n📂 Processing folder: {folder_name}")
        if folder_name not in folder_dict:
            folder_dict[folder_name] = {"folder_name": folder_name}

        # === 每个 folder 对所有任务都跑一下 ===
        for task in task_list:
            print(f"   ➤ Task: {task}")

            try:
                result = analyze_parquet_folder(folder_path, task, model_name)

                # 宽表结构：两个字段
                folder_dict[folder_name][f"{task}_length"] = result.get("avg_token_length", None)
                folder_dict[folder_name][f"{task}_accuracy"] = result.get("avg_extractive_match", None)

            except Exception as e:
                print(f"⚠️ Failed: {folder_name} / {task}: {e}")

                # 即使失败也留空，避免缺列
                folder_dict[folder_name][f"{task}_length"] = None
                folder_dict[folder_name][f"{task}_accuracy"] = None

    # === 输出宽表 CSV ===
    df = pd.DataFrame(folder_dict.values())
    df.to_csv(output_csv, index=False)

    print(f"\n✅ Wide-format summary saved to {output_csv}")
    print(df)
