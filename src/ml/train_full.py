# src/ml/train_full.py

import os
os.system(
    "python src/ml/train.py "
    "--train_path data/raw/train.csv "
    "--test_path data/raw/test.csv "
    "--out_dir outputs/full "
    "--n_splits 5 "
    "--n_rounds 3000 "
    "--early_stopping 150 "
    "--n_hash_features 256 "
    "--seed 42"
)
