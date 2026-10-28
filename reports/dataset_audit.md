# Dataset audit: `data/raw/india_housing_prices.csv`

Inspected on 2026-10-27. File size is 41,113,393 bytes. Parsed the CSV with Python's standard `csv` module; no data-science packages were installed in the environment.

## Dataset summary

- **Shape:** 250,000 rows × 23 columns.
- **Blank fields:** 0 in every column.
- **Exact duplicate rows:** 0.
- **Potential target:** no actual investment, profit, return, appreciation, or future-price outcome column exists. `Price_in_Lakhs` is the current price and `Price_per_SqFt` is a current-price-derived field.
- **Regression:** no defensible five-year future price target is present. This dataset cannot validate future appreciation.
- **Classification:** no observed investment label exists. The implementation may optionally evaluate a proxy label: whether current `Price_per_SqFt` is below its city/property-type peer median. That is a relative current-price proxy only, not profitability, investment suitability, or expected return. The proxy is constructed only for rows with at least two peer observations; any evaluation is about reproducing this proxy.
- **Potential leakage:** exclude `ID`; for the proxy classification also exclude `Price_in_Lakhs` and `Price_per_SqFt`, which define the label. `Age_of_Property` duplicates information derivable from `Year_Built` and the dataset's reference year; retain only a single age representation in predictive inputs.
- **Data quality cautions:** `Locality` values use generic labels such as `Locality_84`; source provenance and semantic validity are not documented. `Age_of_Property` is internally consistent with a 2025 reference year for all 250,000 rows, so it is stale relative to the audit date and should be recalculated from `Year_Built` for a current prediction context. `Floor_No` exceeds `Total_Floors` in 116,304 rows. `Price_per_SqFt` is zero in 2,534 rows and ranges from 0.0 to 0.99. Numeric values parse, but semantic validity and data provenance need confirmation.
- **Proxy class balance (full dataset, descriptive only):** among 250,000 rows across 126 city/property-type groups, 115,671 (46.3%) fall below their peer-group median and 134,329 (53.7%) are at or above it. This is not balance for an observed investment outcome.
- **Sufficiency:** sufficient to demonstrate supervised learning of the clearly defined relative-price proxy and property-price EDA, subject to data provenance concerns. Insufficient to establish investment profitability or train/validate a five-year price forecast.

## Schema and types

All 23 fields are nonblank. Numeric-looking fields are parsed as integers or decimals; remaining fields are categorical/text.

| Column | Observed type | Unique values |
|---|---:|---:|
| ID | integer-like identifier | 250,000 |
| State | categorical | 20 |
| City | categorical | 42 |
| Locality | categorical | 500 |
| Property_Type | categorical | 3 |
| BHK | integer | 5 |
| Size_in_SqFt | integer | 4,501 |
| Price_in_Lakhs | decimal | 48,700 |
| Price_per_SqFt | decimal | 100 |
| Year_Built | integer | 34 |
| Furnished_Status | categorical | 3 |
| Floor_No | integer | 31 |
| Total_Floors | integer | 30 |
| Age_of_Property | integer | 34 |
| Nearby_Schools | integer | 10 |
| Nearby_Hospitals | integer | 10 |
| Public_Transport_Accessibility | categorical | 3 |
| Parking_Space | categorical | 2 |
| Security | categorical | 2 |
| Amenities | delimited categorical text | 325 combinations |
| Facing | categorical | 4 |
| Owner_Type | categorical | 3 |
| Availability_Status | categorical | 2 |

## Numeric ranges observed

- `Price_in_Lakhs`: 10.0–500.0
- `Size_in_SqFt`: 500–5,000
- `Price_per_SqFt`: 0.0–0.99
- `Year_Built`: 1990–2023
- `City`: 42; property types: Villa, Independent House, Apartment

These are observed ranges, not assertions that all values are valid. No provenance, sampling design, or data dictionary was supplied. Before real investment use, verify the source, currency/unit conventions, temporal meaning, and field quality.
