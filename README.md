# Group 21: Anomaly Detection Across Real-World Datasets

Code for the University of Canberra Information & Communication Technology Project (unit 9785, Semester 2 2026), Group 2026-S2-21. The project asks: *How effectively do statistical and machine-learning anomaly detection techniques identify anomalies across different real-world datasets?*

## Repository layout

| Folder | Contents |
|---|---|
| `lof/` | Local Outlier Factor (LOF): `lof_lister.py` runs LOF on one dataset and lists flagged rows; `lof_evaluation.py` reproduces all reported LOF results |
| `data/` | Where the three Kaggle CSV files go (not stored here, see [data/README.md](data/README.md)) |

## Local Outlier Factor (LOF)

### What LOF does

LOF compares how crowded each data point's neighbourhood is with how crowded its neighbours' neighbourhoods are. A score near 1 means the point is as crowded as its neighbours (normal). A score well above 1 means it sits in an emptier area than its neighbours (a likely anomaly). It is unsupervised: the answer labels are hidden from the model and used only afterwards to evaluate it.

### LOF results (k = 20, 5% of rows flagged)

| Dataset | Rows | Anomaly rate | ROC-AUC | Recall | Precision | Precision if random |
|---|---|---|---|---|---|---|
| Financial | 20,000 | 1.7% | 0.790 | 26.3% | 8.9% | 1.7% |
| Network | 9,537 | 44.7% | 0.708 | 8.5% | 75.9% | 44.7% |
| Healthcare | 200,020 | 52.6% | 0.585 | 6.5% | 68.6% | 52.6% |

LOF works best when anomalies are rare and distinct (financial fraud) and is close to random where about half the rows carry the anomaly label (network and healthcare).

### Setup

Requires Python 3.9 or later. Run all commands from the top folder of the repository.

```
pip install -r requirements.txt
```

Download the three datasets from Kaggle into the `data/` folder. See [data/README.md](data/README.md) for links and file names.

### Usage

**Run LOF on one dataset and list the flagged rows**

```
python lof/lof_lister.py data/credit_card_fraud_2026.csv
```

Run it with no file name to choose a CSV in a small window instead. The script prints a confusion matrix, recall, precision and ROC-AUC, and saves the flagged rows next to the dataset as `LOF_flagged_<dataset>.csv`, most suspicious first.

**Reproduce all reported results**

```
python lof/lof_evaluation.py
```

This prints the results table above, plus how results change with the flagging rate and with the neighbourhood size k, and saves them to `results.json`. The healthcare dataset takes several minutes per run.

### Settings

Both are near the top of `lof/lof_lister.py`:

- `n_neighbors` (default 20): neighbourhood size. Very small values (such as 5) react to noise. k = 35 gave a small improvement on the financial and network data.
- `contamination` (default 0.05): share of rows flagged. Higher values catch more anomalies but raise more false alarms.

### Preprocessing

1. Remove rows with genuinely empty values. The text "None" in the network data's `encryption_used` column is kept as its own category, not treated as missing.
2. Hide the label column (`is_fraud`, `attack_detected` or `Risk Category`) from the model.
3. Drop ID columns and text columns where almost every value is unique (such as the healthcare `Timestamp`).
4. One-hot encode categorical columns and standardise all features.

### Known limitations

- The network and healthcare labels cover about half of each dataset, so they are not rare anomalies in the usual sense.
- LOF is slow on large data (about 6 minutes on 200,000 rows).
- The datasets may be synthetic, so results may not carry over to real bank, hospital or network data.
- The model scores the data it was fitted on. Scoring new records as they arrive needs `novelty=True` and a separate training set.

### References

Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying density-based local outliers. *Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data*, 93–104. https://doi.org/10.1145/342009.335388

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
