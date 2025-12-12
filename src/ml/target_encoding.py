# src/ml/target_encoding.py

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def target_encode(train_col, y, test_col, n_splits=5, smoothing=10, seed=42):
    """
    K-fold target encoding with smoothing.
    Returns encoded_train, encoded_test
    """
    oof = np.zeros(len(train_col))
    test_enc = np.zeros(len(test_col))
    
    global_mean = y.mean()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for tr, va in kf.split(train_col):
        tr_mean = train_col.iloc[tr].to_frame().join(y.iloc[tr]).groupby(train_col.name)["target"].mean()
        oof[va] = train_col.iloc[va].map(lambda x: (smoothing*global_mean + tr_mean.get(x, global_mean)) / (smoothing + 1))

    # full fit
    full_mean = train_col.to_frame().join(y).groupby(train_col.name)["target"].mean()
    test_enc = test_col.map(lambda x: (smoothing*global_mean + full_mean.get(x, global_mean)) / (smoothing + 1))

    return oof, test_enc
