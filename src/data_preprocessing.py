"""Leakage-aware feature and target preparation."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import model_features


def make_preprocessor(features):
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]), categorical))
    if not transformers:
        raise ValueError("No usable features remain after excluding identifiers and price-derived columns.")
    return ColumnTransformer(transformers, remainder="drop")


def relative_value_target(frame):
    """Proxy: below peer median current price/sqft; never actual profitability."""
    required = {"City", "Property_Type", "Price_per_SqFt"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Relative-value proxy needs columns: {sorted(required)}")
    price_sqft = pd.to_numeric(frame["Price_per_SqFt"], errors="coerce")
    groups = frame["City"].astype("string") + "|" + frame["Property_Type"].astype("string")
    counts = groups.groupby(groups).transform("size")
    medians = price_sqft.groupby(groups).transform("median")
    valid = price_sqft.notna() & medians.notna() & (counts >= 2)
    y = (price_sqft < medians).astype("int8")
    return y.loc[valid], valid


def prepare_classification(frame):
    y, valid = relative_value_target(frame)
    X = model_features(frame).loc[valid]
    return X, y
