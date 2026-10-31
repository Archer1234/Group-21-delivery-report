import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
import tkinter as tk
from tkinter import filedialog, messagebox

# Settings (change these to experiment)
n_neighbors = 20
contamination = 0.05  # share of rows LOF will flag (5%)

def run_lof(filepath):
    print("Loading dataset...")
    # Only treat empty cells as missing, so "None" in encryption_used stays as its own category
    df = pd.read_csv(filepath, keep_default_na=False, na_values=[""])
    df_clean = df.dropna().copy()
    print(f"Rows loaded: {len(df)}, rows after removing missing values: {len(df_clean)}")

    # 1. Auto-detect the 'Answer Key' column based on which CSV is loaded
    target_col = None
    anomaly_label = 1 # Default for fraud and cyber attacks

    if 'is_fraud' in df_clean.columns:
        target_col = 'is_fraud'
    elif 'attack_detected' in df_clean.columns:
        target_col = 'attack_detected'
    elif 'Risk Category' in df_clean.columns:
        target_col = 'Risk Category'
        anomaly_label = 'High Risk' # Healthcare uses text instead of a 1

    if target_col:
        print(f"Detected evaluation column: {target_col}")

    # 2. Automatically remove ID columns (transaction_id, session_id, Patient ID)
    cols_to_drop = [col for col in df_clean.columns if 'id' in col.lower()]

    # Also remove text columns where nearly every value is different (like Timestamp)
    # otherwise encoding them makes thousands of new columns and runs out of memory
    for col in df_clean.select_dtypes(include=['object', 'string']).columns:
        if col != target_col and col not in cols_to_drop and df_clean[col].nunique() > 50:
            cols_to_drop.append(col)

    print(f"Ignoring columns: {cols_to_drop}")

    # 3. Exclude the answer key from the training data
    df_model = df_clean.drop(columns=cols_to_drop + ([target_col] if target_col else []))

    print("Encoding text and standardising numbers...")
    text_cols = df_model.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    df_encoded = pd.get_dummies(df_model, columns=text_cols, drop_first=True).astype(float)

    # Scale every column (including True/False ones) so no column dominates the distances
    scaler = StandardScaler()
    df_encoded[df_encoded.columns] = scaler.fit_transform(df_encoded)

    print("Running the LOF model on the full data...")
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    df_clean['LOF_Anomaly_Flag'] = lof.fit_predict(df_encoded)  # -1 = anomaly, 1 = normal
    df_clean['LOF_Score'] = -lof.negative_outlier_factor_      # higher = more unusual

    # 4. Universal Results Validation
    results = ""
    if target_col:
        actual = (df_clean[target_col] == anomaly_label).astype(int)
        predicted = (df_clean['LOF_Anomaly_Flag'] == -1).astype(int)
        tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
        random_rate = actual.mean()

        results = "\n--- RESULTS VALIDATION ---\n"
        results += f"Total actual anomalies in dataset: {actual.sum()} ({random_rate:.1%} of rows)\n"
        results += f"Rows flagged by LOF: {predicted.sum()}\n\n"
        results += f"Caught (true positives): {tp}\n"
        results += f"Missed (false negatives): {fn}\n"
        results += f"False alarms (false positives): {fp}\n"
        results += f"Correctly ignored (true negatives): {tn}\n\n"
        results += f"Recall (share of real anomalies caught): {recall_score(actual, predicted):.1%}\n"
        results += f"Precision (share of flags that were real): {precision_score(actual, predicted, zero_division=0):.1%}\n"
        results += f"ROC-AUC (0.5 = random guessing, 1.0 = perfect): {roc_auc_score(actual, df_clean['LOF_Score']):.3f}\n\n"
        results += f"Random guessing would get: recall {contamination:.1%}, precision {random_rate:.1%}, AUC 0.500"
        print(results)

    print("\nExtracting flagged records...")
    flagged_list = df_clean[df_clean['LOF_Anomaly_Flag'] == -1].sort_values('LOF_Score', ascending=False)

    # Name the output after the dataset so runs don't overwrite each other
    dataset_name = os.path.splitext(os.path.basename(filepath))[0]
    output_filename = os.path.join(os.path.dirname(filepath), f"LOF_flagged_{dataset_name}.csv")
    flagged_list.to_csv(output_filename, index=False)
    print(f"List generated! Saved {len(flagged_list)} suspicious records to {output_filename}")

    return results, output_filename, len(flagged_list)

def browse_and_run():
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not filepath:
        return

    try:
        results, output_filename, flagged_count = run_lof(filepath)
        messagebox.showinfo("Success", f"Analysis complete!\n{flagged_count} suspicious records saved to {output_filename}\n{results}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Run from the command line if a CSV is given (python lof_lister.py data/file.csv)
if __name__ == "__main__" and len(sys.argv) > 1:
    for path in sys.argv[1:]:
        run_lof(path)

# Otherwise open the window
elif __name__ == "__main__":
    # --- Simplified UI ---
    root = tk.Tk()
    root.title("LOF Universal Anomaly Lister")
    root.geometry("400x200")

    tk.Label(root, text="LOF Anomaly List Generator", font=("Arial", 14, "bold")).pack(pady=20)
    tk.Label(root, text="Click below to select your CSV.\nThe program will generate a list of flagged anomalies.", font=("Arial", 10)).pack(pady=10)
    tk.Button(root, text="Select CSV and Generate List", command=browse_and_run, bg="green", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
    root.mainloop()
