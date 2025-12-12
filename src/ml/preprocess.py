# src/ml/preprocess.py
"""
Tabular preprocessing for System Thread Forecaster.

Provides TabularPreprocessor which:
 - auto-detects numeric / low-card / high-card categorical columns
 - extracts date features from DateAS, DateOS
 - numeric: median impute + StandardScaler
 - low-card categorical: constant impute + OneHotEncoder
 - high-card categorical: FeatureHasher (n_features configurable)
 - exposes fit/transform/fit_transform/get_feature_names_out/save/load
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction import FeatureHasher
import joblib
import os

# ---------- helpers ----------
def _to_dicts_for_hasher(X):
    """
    Convert pandas DataFrame or 2D-array slice to list-of-dicts for FeatureHasher.
    Keys are column names if available, else 'c0','c1',...
    """
    if hasattr(X, "to_numpy") and hasattr(X, "columns"):
        df = X.fillna("__missing__").astype(str)
        return df.to_dict(orient="records")
    # fallback for numpy
    return [{f"c{i}": str(v) for i, v in enumerate(row)} for row in X]

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract simple date features from provided date columns.
    Adds: <col>_year, <col>_month, <col>_day, <col>_age_days
    If date is missing, fills with 0 (year/month/day) and large age value.
    """
    def __init__(self, date_cols: Optional[List[str]] = None):
        self.date_cols = date_cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X2 = X.copy()
        for c in self.date_cols:
            if c in X2.columns:
                dt = pd.to_datetime(X2[c], errors="coerce")
                X2[f"{c}_year"] = dt.dt.year.fillna(0).astype(int)
                X2[f"{c}_month"] = dt.dt.month.fillna(0).astype(int)
                X2[f"{c}_day"] = dt.dt.day.fillna(0).astype(int)
                maxd = dt.max()
                if pd.isna(maxd):
                    X2[f"{c}_age_days"] = 999999
                else:
                    X2[f"{c}_age_days"] = (maxd - dt).dt.days.fillna(999999).astype(int)
        # Drop original date columns so downstream doesn't duplicate
        for c in self.date_cols:
            if c in X2.columns:
                X2 = X2.drop(columns=[c])
        return X2

# ---------- main preprocessor class ----------
class TabularPreprocessor:
    """
    Build and manage a preprocessing pipeline for tabular data.
    Methods:
      - fit(df)
      - transform(df) -> np.ndarray
      - fit_transform(df) -> np.ndarray
      - get_feature_names_out() -> List[str]
      - save(path) / load(path)
    """
    def __init__(self,
                 n_hash_features: int = 64,
                 low_card_thresh: int = 30,
                 drop_id_cols: Optional[List[str]] = None,
                 date_cols: Optional[List[str]] = None,
                 target_col: str = "target"):
        self.n_hash_features = int(n_hash_features)
        self.low_card_thresh = int(low_card_thresh)
        self.drop_id_cols = drop_id_cols or ["MachineID"]
        self.date_cols = date_cols or ["DateAS", "DateOS"]
        self.target_col = target_col

        # attributes set on fit
        self.num_cols: List[str] = []
        self.low_card_cols: List[str] = []
        self.high_card_cols: List[str] = []
        self.preprocessor: Optional[ColumnTransformer] = None
        self.full_pipeline: Optional[Pipeline] = None
        self.ohe_categories_: Optional[Dict[str, List[str]]] = None
        self.fitted = False
        self.hasher_feature_names_: List[str] = []
        self.date_cols_present_: List[str] = []

    def _detect_columns(self, df: pd.DataFrame):
        """
        Detect numeric, low-cardinality and high-cardinality categorical columns,
        excluding identifier, target and date columns so ColumnTransformer later
        does not expect columns that DateFeatureExtractor will remove.
        """
        # Work on a copy and drop id/target cols
        candidate = df.copy()
        for c in [self.target_col] + self.drop_id_cols:
            if c in candidate.columns:
                candidate = candidate.drop(columns=[c])

        # ALSO drop date columns from candidate so they are not treated as features
        date_cols_present = [c for c in self.date_cols if c in candidate.columns]
        if date_cols_present:
            candidate = candidate.drop(columns=date_cols_present)

        # Object columns (categorical)
        obj_cols = candidate.select_dtypes(include=["object"]).columns.tolist()
        # Numeric columns (int/float)
        num_cols = candidate.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # Low vs high cardinality split for object columns
        low_card = [c for c in obj_cols if candidate[c].nunique(dropna=False) < self.low_card_thresh]
        high_card = [c for c in obj_cols if candidate[c].nunique(dropna=False) >= self.low_card_thresh]

        # store detected lists
        self.num_cols = num_cols
        self.low_card_cols = low_card
        self.high_card_cols = high_card
        # also store which date cols are present to use later
        self.date_cols_present_ = date_cols_present

    def fit(self, df: pd.DataFrame):
        """
        Fit the preprocessing pipeline on the training dataframe.
        """
        # 1. preliminary column detection
        self._detect_columns(df)

        # 2. create sub-pipelines
        numeric_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # compatibility for OneHotEncoder across sklearn versions
        def _get_onehot_encoder():
            try:
                # old sklearn
                return OneHotEncoder(handle_unknown="ignore", sparse=False)
            except TypeError:
                # newer sklearn
                return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        cat_low_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("ohe", _get_onehot_encoder())
        ])

        high_pipe = Pipeline(steps=[
            ("to_dict", FunctionTransformer(_to_dicts_for_hasher, validate=False)),
            ("hasher", FeatureHasher(n_features=self.n_hash_features, input_type="dict"))
        ])

        transformers = []
        if len(self.num_cols) > 0:
            transformers.append(("num", numeric_pipe, self.num_cols))
        if len(self.low_card_cols) > 0:
            transformers.append(("cat_low", cat_low_pipe, self.low_card_cols))
        if len(self.high_card_cols) > 0:
            transformers.append(("cat_high", high_pipe, self.high_card_cols))

        # Build ColumnTransformer
        self.preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)

        # Full pipeline: date extractor then column transformer
        date_cols_present = [c for c in self.date_cols if c in df.columns]
        self.full_pipeline = Pipeline(steps=[
            ("date_feats", DateFeatureExtractor(date_cols=date_cols_present)),
            ("col_trans", self.preprocessor)
        ])

        # Fit the pipeline
        # Drop id columns & target before fitting
        df_fit = df.copy()
        for c in self.drop_id_cols + [self.target_col]:
            if c in df_fit.columns:
                df_fit = df_fit.drop(columns=[c])
        # Note: date columns are kept here because DateFeatureExtractor needs them
        self.full_pipeline.fit(df_fit)

        # capture onehot categories (for feature names)
        self.ohe_categories_ = {}
        if "cat_low" in dict(self.preprocessor.named_transformers_):
            ohe = self.preprocessor.named_transformers_["cat_low"].named_steps["ohe"]
            # onehot encoder categories_ is in same order as low_card_cols
            for col, cats in zip(self.low_card_cols, ohe.categories_):
                self.ohe_categories_[col] = list(map(str, cats))

        # prepare hashed feature names
        self.hasher_feature_names_ = [f"hash_{i}" for i in range(self.n_hash_features)] if len(self.high_card_cols) > 0 else []

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform dataframe into feature matrix (numpy array).
        Use after fit().
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform(). Call fit(df) first.")

        # Drop ID & target if present (keeps transform safe for train/test)
        df_t = df.copy()
        for c in self.drop_id_cols + [self.target_col]:
            if c in df_t.columns:
                df_t = df_t.drop(columns=[c])

        X_trans = self.full_pipeline.transform(df_t)

        # ColumnTransformer + FeatureHasher may return sparse matrix or ndarray
        if hasattr(X_trans, "toarray"):
            X_arr = X_trans.toarray()
        else:
            X_arr = np.asarray(X_trans)

        return X_arr

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def get_feature_names_out(self) -> List[str]:
        """
        Return list of feature names in the same order as transform() output columns.
        Order produced by pipeline: date-derived columns (for each date col: year, month, day, age_days),
        then ColumnTransformer outputs in the order transformers were added:
            - numeric columns (original names)
            - low-card OHE columns (col=category)
            - high-card hashed columns (hash_0..hash_n-1)
        """
        if not self.fitted:
            raise RuntimeError("fit() must be called before get_feature_names_out()")

        feat_names: List[str] = []

        # Date-derived names (only include those date cols that were present during fit)
        for c in self.date_cols_present_:
            feat_names.extend([f"{c}_year", f"{c}_month", f"{c}_day", f"{c}_age_days"])

        # Now columntransformer parts in same order as transformers list
        # Numeric
        feat_names.extend(self.num_cols)

        # Low-card OHE names (use stored categories)
        if self.ohe_categories_:
            for col in self.low_card_cols:
                cats = self.ohe_categories_.get(col, [])
                # OneHotEncoder with handle_unknown='ignore' includes all categories
                for cat in cats:
                    feat_names.append(f"{col}={cat}")

        # High-card hashed feature names
        feat_names.extend(self.hasher_feature_names_)

        return feat_names

    def save(self, path: str):
        """
        Save the fitted preprocessor to disk (joblib).
        """
        if not self.fitted:
            raise RuntimeError("fit() must be called before save()")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "TabularPreprocessor":
        """
        Load preprocessor instance from disk (joblib).
        """
        return joblib.load(path)


# ---------- quick usage helpers ----------
def build_preprocessor_from_df(df: pd.DataFrame,
                               n_hash_features: int = 64,
                               low_card_thresh: int = 30,
                               drop_id_cols: Optional[List[str]] = None,
                               date_cols: Optional[List[str]] = None,
                               target_col: str = "target") -> Tuple[TabularPreprocessor, Dict]:
    """
    Convenience to quickly build and fit a TabularPreprocessor on a sample dataframe.
    Returns (preprocessor_instance, feature_lists_dict)
    """
    tp = TabularPreprocessor(n_hash_features=n_hash_features,
                             low_card_thresh=low_card_thresh,
                             drop_id_cols=drop_id_cols,
                             date_cols=date_cols,
                             target_col=target_col)
    tp.fit(df)
    feature_lists = {
        "num_cols": tp.num_cols,
        "low_card_cols": tp.low_card_cols,
        "high_card_cols": tp.high_card_cols,
        "date_cols": tp.date_cols_present_
    }
    return tp, feature_lists
