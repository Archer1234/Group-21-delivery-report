import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score

# The three datasets (put the CSVs in the data folder)
datasets = {
    "Financial": "credit_card_fraud_2026.csv",
    "Network": "cybersecurity_intrusion_data.csv",
    "Healthcare": "human_vital_signs_dataset_2024.csv",
}

# Settings to test
contamination_values = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
k_values = [5, 10, 20, 35, 50, 100, 200]

def prepare_data(filepath):
    """Cleans the data the same way as lof_lister.py and returns the features and answers"""
    df = pd.read_csv(filepath, keep_default_na=False, na_values=[""]).dropna()

    # Find the answer key column
    if 'is_fraud' in df.columns:
        target_col, anomaly_label = 'is_fraud', 1
    elif 'attack_detected' in df.columns:
        target_col, anomaly_label = 'attack_detected', 1
    else:
        target_col, anomaly_label = 'Risk Category', 'High Risk'

    # Drop ID columns and text columns where nearly every value is different
    cols_to_drop = [col for col in df.columns if 'id' in col.lower()]
    for col in df.select_dtypes(include=['object', 'string']).columns:
        if col != target_col and col not in cols_to_drop and df[col].nunique() > 50:
            cols_to_drop.append(col)

    # Encode text columns and scale everything
    df_model = df.drop(columns=cols_to_drop + [target_col])
    text_cols = df_model.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    df_encoded = pd.get_dummies(df_model, columns=text_cols, drop_first=True).astype(float)
    df_encoded[df_encoded.columns] = StandardScaler().fit_transform(df_encoded)

    answers = (df[target_col] == anomaly_label).astype(int).values
    return df_encoded.values, answers

def get_lof_scores(features, k):
    """Runs LOF and returns a score for every row (higher = more unusual)"""
    lof = LocalOutlierFactor(n_neighbors=k)
    lof.fit(features)
    return -lof.negative_outlier_factor_

def check_flagged(scores, answers, flag_rate):
    """Flags the top share of scores and works out recall and precision"""
    flagged = scores > np.quantile(scores, 1 - flag_rate)
    caught = int((flagged & (answers == 1)).sum())
    false_alarms = int((flagged & (answers == 0)).sum())
    return {
        "recall": caught / answers.sum(),
        "precision": caught / max(flagged.sum(), 1),
        "caught": caught,
        "false_alarms": false_alarms,
    }

def main():
    all_results = {}

    for name, filename in datasets.items():
        filepath = os.path.join("data", filename)
        if not os.path.exists(filepath):
            print(f"Skipping {name}: {filepath} not found")
            continue

        print(f"\nRunning LOF on {name}...")
        features, answers = prepare_data(filepath)

        # 1. Main run with k = 20
        start = time.time()
        scores = get_lof_scores(features, 20)
        results = {
            "rows": len(answers),
            "anomaly_rate": answers.mean(),
            "roc_auc": roc_auc_score(answers, scores),
            "seconds": time.time() - start,
        }

        # 2. Try different flagging rates (only changes the cut-off, not the scores)
        results["by_contamination"] = {rate: check_flagged(scores, answers, rate) for rate in contamination_values}

        # 3. Try different neighbourhood sizes (skip healthcare as 200k rows takes too long)
        if name != "Healthcare":
            results["auc_by_k"] = {k: roc_auc_score(answers, get_lof_scores(features, k)) for k in k_values}

        all_results[name] = results

        # Print a summary
        at_5 = results["by_contamination"][0.05]
        print(f"{results['rows']} rows, {results['anomaly_rate']:.1%} anomalies")
        print(f"ROC-AUC (k=20): {results['roc_auc']:.3f}")
        print(f"At 5% flagged: recall {at_5['recall']:.1%}, precision {at_5['precision']:.1%} (random guessing: {results['anomaly_rate']:.1%})")
        if "auc_by_k" in results:
            print("AUC by k: " + ", ".join(f"k={k}: {auc:.3f}" for k, auc in results["auc_by_k"].items()))

    # Save everything
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=1, default=float)
    print("\nResults saved to results.json")

if __name__ == "__main__":
    main()
