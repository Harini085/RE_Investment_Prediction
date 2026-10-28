# Real Estate Investment Advisor

An end-to-end MVP for exploring the supplied India housing listings, estimating a **relative current-price proxy**, and calculating a transparent five-year appreciation scenario. The supplied data has no observed investment outcome or future price; this project does not claim to predict profitability or future value. See [the dataset audit](reports/dataset_audit.md) for field-by-field inspection and caveats.

## Data and prediction definitions

Dataset: `data/raw/india_housing_prices.csv` (250,000 records, 23 columns). It has current `Price_in_Lakhs` and `Price_per_SqFt`, but no genuine investment label, historical appreciation, or five-year future price. Data provenance was not supplied; generic `Locality_N` values and some inconsistent floor/age fields deserve validation before real-world use.

The optional classifier label is **below the median current price per square foot among rows sharing City and Property_Type**. It is a transparent descriptive proxy, not evidence that a property is profitable, undervalued, or suitable. Current price and price per square foot are excluded from its model inputs to prevent direct target leakage. IDs are excluded too. The held-out metrics measure prediction of this constructed proxy only. The regression training command records that no valid future target is available and skips model training. The app's five-year calculator applies `current price × (1 + assumed annual rate)^years`; it is a scenario, not a forecast.

## Windows PowerShell setup

```powershell
cd "C:\Users\91915\Downloads\Labmentix\RE_Investment_Prediction"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The execution environment used for this implementation reports Python 3.14.7, but its `venv` creation failed while bootstrapping `ensurepip`; the resulting `.venv` has no pip. Install or select a Python distribution with working `pip`/`ensurepip`, recreate the virtual environment, then run the setup commands above. No dependencies were installed in this environment.

## Run

```powershell
python -m src.eda
python -m src.train_classification
python -m src.train_regression
streamlit run app.py
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --workers 1
```

The dashboard is at `http://localhost:8501`; MLflow UI is at `http://127.0.0.1:5000`. MLflow metadata is stored in the project's `mlflow.db` SQLite database and local model artifacts are stored under `mlartifacts/`. Training code resolves the database path from the project root. Launch the UI command from the project root so its relative SQLite URI points to that same database. Current MLflow releases place SQLite as the default local backend and treat the filesystem store as legacy; see the [MLflow backend-store guide](https://mlflow.org/docs/latest/tracking/backend-stores/) and [file-store migration guide](https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store/).

## Architecture

- `src/utils.py`: paths, loading, deterministic cleaning, amenity count, feature availability, and local MLflow setup.
- `src/data_preprocessing.py`: train-fitted imputation, scaling, one-hot encoding with unknown-category support, and proxy target creation.
- `src/eda.py`: actual dataset summary and saved distributions, price/size scatter, city prices, and numeric correlation chart.
- `src/train_classification.py`: stratified train/validation/test split, Logistic Regression and Random Forest pipelines, accuracy/precision/recall/F1, confusion matrices, candidate and selected MLflow runs, comparison report, and saved complete pipeline. Selection uses validation F1 and favors Logistic Regression when it is within 0.01 F1 of the best score to keep the saved artifact compact.
- `src/train_regression.py`: checks for a real future-price field, logs a skipped run when absent, and does not substitute current price.
- `app.py`: filtered EDA, dynamic property inputs, proxy classifier (when trained), evaluation metrics (when available), and configurable five-year scenario.

Generated outputs are written to `reports/` and the fitted pipeline to `models/`. MLflow tracking metadata is stored in `mlflow.db` and artifacts in `mlartifacts/`. Generated files are ignored by Git. Metrics are populated only after training succeeds; none are hardcoded.

### Latest successful classification run

Candidate validation results: Logistic Regression — accuracy 0.6977, precision 0.6652, recall 0.6979, F1 0.6812; Random Forest — accuracy 0.6957, precision 0.6538, recall 0.7276, F1 0.6888. The Random Forest validation F1 gain was under the documented 0.01 threshold, so the smaller Logistic Regression pipeline was selected. On its untouched test split, it achieved accuracy 0.6971, precision 0.6631, recall 0.7020, and F1 0.6820. These metrics predict only the relative-price proxy described above.

## Limitations and next steps

The file's source, collection method, units, and intended population are undocumented. `Age_of_Property` is fixed to a 2025 reference year and the floor fields conflict in 116,304 rows; verify or clean these before relying on model outputs. The proxy reflects relative asking-price-per-area within city/property-type groups, not returns. Random row splitting may overstate generalization across repeated localities or source patterns; a future version should validate provenance and consider locality- or time-based holdouts if the data supports them. Five-year scenarios assume a constant user-entered appreciation rate and omit uncertainty, costs, financing, tax, inflation, rent, and liquidity. Outputs are estimates for demonstration and are not financial advice.
