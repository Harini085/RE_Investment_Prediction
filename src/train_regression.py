"""Record regression target availability; never substitute current price for future price."""
from .utils import REPORT_DIR, get_mlflow, load_data


FUTURE_TARGET_CANDIDATES = {
    "Price_After_5_Years", "Price_in_5_Years", "Future_Price_in_Lakhs",
    "Price_5Y_Later", "Price_After_Five_Years", "Future_Price",
}


def train():
    mlflow = get_mlflow()
    data = load_data()
    target = next((column for column in FUTURE_TARGET_CANDIDATES if column in data.columns), None)
    with mlflow.start_run(run_name="regression-target-audit"):
        mlflow.log_param("experiment_type", "five_year_future_price_regression")
        mlflow.log_param("dataset_rows", len(data))
        mlflow.log_param("target_available", bool(target))
        if target is None:
            mlflow.set_tag("status", "skipped_no_valid_future_target")
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            note = REPORT_DIR / "regression_target_audit.txt"
            note.write_text("No genuine five-year future price field is present. No regression metrics or model were created. Use the app's clearly labeled appreciation scenario calculator.\n", encoding="utf-8")
            mlflow.log_artifact(str(note), artifact_path="target_audit")
            print("Regression skipped: dataset contains no valid five-year future-price target.")
            return None
    # A future-named column alone is not proof of provenance; require verification before fitting.
    raise NotImplementedError(f"Found candidate {target!r}; verify its provenance and observation horizon before using it as a target.")


if __name__ == "__main__":
    train()
