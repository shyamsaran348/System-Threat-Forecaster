# src/ml/evaluate_oof.py

import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt

oof = pd.read_csv("outputs/metrics/oof_predictions.csv")
y = oof["target"].values
pred = oof["oof_pred"].values

print("OOF AUC =", roc_auc_score(y, pred))

# PR curve
prec, rec, thr = precision_recall_curve(y, pred)
plt.plot(rec, prec)
plt.title("Precision–Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# threshold @ 0.5
cm = confusion_matrix(y, pred > 0.5)
print("Confusion Matrix @0.5:")
print(cm)
