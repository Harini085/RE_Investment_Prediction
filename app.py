"""Single-page dashboard for observed housing data and limited model outputs."""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.utils import DATA_PATH, MODEL_DIR, load_data, model_features  # noqa: E402

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")
st.title("Real Estate Investment Advisor")
st.caption("Explore listed housing data, a relative-price proxy model, and an assumption-based five-year scenario.")
st.warning("This dashboard is an educational data-science demonstration, not financial or investment advice. The model target is a relative current-price proxy, not profitability. Scenario values are not validated forecasts and do not guarantee outcomes.")


@st.cache_data(show_spinner="Loading housing dataset…")
def get_data():
    return load_data()


@st.cache_resource
def get_bundle():
    path = MODEL_DIR / "classification_pipeline.joblib"
    return joblib.load(path) if path.exists() else None


try:
    df = get_data()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.info("Place `india_housing_prices.csv` in `data/raw/`, then refresh this page.")
    st.stop()

st.subheader("Dataset overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{len(df):,}"); c2.metric("Columns", len(df.columns)); c3.metric("Exact duplicate rows", f"{df.duplicated().sum():,}"); c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
st.caption(f"Source file: `{DATA_PATH.relative_to(ROOT)}`. Fields describe listed properties; source provenance was not included with the file.")

st.divider()
st.subheader("Explore listings")
filtered = df.copy()
filter_cols = st.columns(4)
for column_index, (col, label) in enumerate([("City", "City"), ("Property_Type", "Property type"), ("BHK", "BHK")]):
    if col in filtered:
        vals = sorted(filtered[col].dropna().unique().tolist(), key=str)
        chosen = filter_cols[column_index].multiselect(label, vals, default=[])
        if chosen:
            filtered = filtered[filtered[col].isin(chosen)]
if "Price_in_Lakhs" in filtered:
    lo, hi = float(filtered.Price_in_Lakhs.min()), float(filtered.Price_in_Lakhs.max())
    price_range = filter_cols[3].slider("Price (lakhs)", lo, hi, (lo, hi))
    filtered = filtered[filtered.Price_in_Lakhs.between(*price_range)]
st.write(f"Showing {len(filtered):,} matching rows.")
if "Price_in_Lakhs" in filtered:
    st.bar_chart(filtered.groupby("City").Price_in_Lakhs.median().sort_values(ascending=False).head(15), y_label="Median price (lakhs)")
if {"Size_in_SqFt", "Price_in_Lakhs"}.issubset(filtered):
    st.scatter_chart(filtered.sample(min(5000, len(filtered)), random_state=42), x="Size_in_SqFt", y="Price_in_Lakhs", color="Property_Type" if "Property_Type" in filtered else None)

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Relative-value proxy (classification)")
    st.write("The classifier estimates whether a listing is below the observed median current price per square foot among listings in the same city and property type. This is only a relative-price proxy; it does not measure profit, future returns, or suitability.")
    bundle = get_bundle()
    if bundle is None:
        st.info("No trained classifier is available yet. Run `python -m src.train_classification` after installing requirements.")
    else:
        feat_names = bundle["features"]
        form_sample = df.sample(min(5000, len(df)), random_state=42)
        feature_data = model_features(form_sample).reindex(columns=feat_names)
        with st.form("prediction_form"):
            values = {}
            for feature in feat_names:
                series = feature_data[feature]
                if pd.api.types.is_numeric_dtype(series):
                    val = pd.to_numeric(series, errors="coerce").median()
                    values[feature] = st.number_input(feature.replace("_", " "), min_value=0.0,
                        value=max(0.0, float(val)) if pd.notna(val) else 0.0)
                else:
                    choices = sorted(series.dropna().astype(str).unique().tolist())[:100]
                    if not choices: choices = ["Unknown"]
                    values[feature] = st.selectbox(feature.replace("_", " "), choices)
            submit = st.form_submit_button("Predict Investment Suitability")
        if submit:
            row = pd.DataFrame([values], columns=feat_names)
            try:
                if values.get("Size_in_SqFt", 1) <= 0:
                    raise ValueError("Property size must be greater than zero.")
                if values.get("BHK", 1) <= 0:
                    raise ValueError("BHK must be greater than zero.")
                if values.get("Floor_No", 0) > values.get("Total_Floors", float("inf")):
                    raise ValueError("Floor number cannot exceed the total floors in the building.")
                pred = int(bundle["pipeline"].predict(row)[0])
                label = "Below peer median current price per sq ft" if pred == 1 else "At or above peer median current price per sq ft"
                st.success(f"Relative-value proxy result: **{label}**")
                model = bundle["pipeline"].named_steps["model"]
                if hasattr(model, "predict_proba"):
                    probability = bundle["pipeline"].predict_proba(row)[0][pred]
                    st.caption(f"Model probability for this proxy label: {probability:.1%}. This is not a probability of profit or investment success.")
            except Exception as exc:
                st.error(f"Input could not be scored: {exc}")

with right:
    st.subheader("Five-year value scenario")
    st.write("No genuine future price target is available. Enter a current price and an assumed annual appreciation rate to calculate a scenario, not an ML forecast.")
    with st.form("scenario_form"):
        current_price = st.number_input("Current property price (lakhs)", min_value=0.01, value=100.0, step=1.0)
        annual_rate = st.number_input("Assumed annual appreciation (%)", min_value=-99.0, max_value=100.0, value=5.0, step=0.5)
        years = st.number_input("Years", min_value=1, max_value=30, value=5)
        scenario = st.form_submit_button("Calculate scenario")
    if scenario:
        future = current_price * (1 + annual_rate / 100) ** years
        st.metric(f"Scenario value after {years} years (lakhs)", f"{future:,.2f}")
        st.caption(f"Formula: {current_price:.2f} × (1 + {annual_rate:.2f}% )^{years}. Assumes a constant annual rate and excludes costs, inflation, taxes, and uncertainty.")

st.divider()
st.subheader("Model evaluation")
metrics_path = ROOT / "reports" / "classification_metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.caption("Held-out metrics measure prediction of the relative-value proxy only. They do not measure investment outcomes.")
    st.caption("Candidate comparison on validation split (selection uses validation F1):")
    st.dataframe(pd.DataFrame(metrics["results"]).drop(columns=["run_id"], errors="ignore"), use_container_width=True, hide_index=True)
    st.write(f"Selected model: **{metrics['selected_model']}**")
    st.caption("Final untouched test metrics for the selected model:")
    st.dataframe(pd.DataFrame([metrics.get("final_test_metrics", {})]), use_container_width=True, hide_index=True)
    st.caption("Class balance (proxy labels): " + ", ".join(f"{k}: {v:,}" for k, v in metrics.get("class_balance", {}).items()))
else:
    st.info("No evaluated classifier metrics yet. Run the classification training script to generate actual held-out results and MLflow runs.")
reg_path = ROOT / "reports" / "figures" / "feature_importance.png"
if reg_path.exists(): st.image(str(reg_path), caption="Model feature importance")
cm_path = ROOT / "reports" / "figures" / "confusion_matrix_selected_test.png"
if cm_path.exists(): st.image(str(cm_path), caption="Selected classifier: untouched test confusion matrix")
st.caption("No five-year price regression is reported because the supplied data has no verified future-price target.")
