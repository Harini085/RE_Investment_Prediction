"""Dataset audit charts and concise descriptive summary."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import DATA_PATH, REPORT_DIR, ROOT, load_data, clean_data


def run_eda():
    data = clean_data(load_data())
    out = REPORT_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": DATA_PATH.relative_to(ROOT).as_posix(),
        "rows": int(len(data)), "columns": int(data.shape[1]),
        "exact_duplicate_rows": int(load_data().duplicated().sum()),
        "missing_by_column": {k: int(v) for k, v in data.isna().sum().items()},
        "price_lakhs_median": float(data["Price_in_Lakhs"].median()) if "Price_in_Lakhs" in data else None,
        "price_lakhs_range": [float(data["Price_in_Lakhs"].min()), float(data["Price_in_Lakhs"].max())] if "Price_in_Lakhs" in data else None,
        "cities": int(data["City"].nunique()) if "City" in data else None,
    }
    if "Price_in_Lakhs" in data:
        fig, ax = plt.subplots(figsize=(8, 5)); sns.histplot(data=data, x="Price_in_Lakhs", bins=40, ax=ax); ax.set(title="Current listed price distribution", xlabel="Price (lakhs)"); fig.tight_layout(); fig.savefig(out / "price_distribution.png", dpi=140); plt.close(fig)
    if "Size_in_SqFt" in data:
        fig, ax = plt.subplots(figsize=(8, 5)); sns.histplot(data=data, x="Size_in_SqFt", bins=40, ax=ax); ax.set(title="Property size distribution", xlabel="Size (sq ft)"); fig.tight_layout(); fig.savefig(out / "size_distribution.png", dpi=140); plt.close(fig)
    if {"Size_in_SqFt", "Price_in_Lakhs"}.issubset(data):
        sample = data.sample(min(10000, len(data)), random_state=42)
        fig, ax = plt.subplots(figsize=(8, 5)); sns.scatterplot(data=sample, x="Size_in_SqFt", y="Price_in_Lakhs", hue="Property_Type" if "Property_Type" in data else None, alpha=.45, s=16, ax=ax); ax.set(title="Listed price versus size", xlabel="Size (sq ft)", ylabel="Price (lakhs)"); fig.tight_layout(); fig.savefig(out / "price_vs_size.png", dpi=140); plt.close(fig)
    if "City" in data and "Price_in_Lakhs" in data:
        top = data.groupby("City")["Price_in_Lakhs"].median().nlargest(15).sort_values()
        fig, ax = plt.subplots(figsize=(9, 6)); top.plot.barh(ax=ax); ax.set(title="Median listed price by city (15 highest)", xlabel="Price (lakhs)"); fig.tight_layout(); fig.savefig(out / "median_price_by_city.png", dpi=140); plt.close(fig)
    numeric = data.select_dtypes(include="number").drop(columns=[c for c in ["ID"] if c in data], errors="ignore")
    if len(numeric.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8)); sns.heatmap(numeric.corr(), cmap="vlag", center=0, ax=ax); ax.set_title("Numeric feature correlation"); fig.tight_layout(); fig.savefig(out / "correlation_heatmap.png", dpi=140); plt.close(fig)
    (REPORT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_eda()
