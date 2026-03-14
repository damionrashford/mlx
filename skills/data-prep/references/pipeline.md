# Data Preparation Pipeline

Full pipeline from raw data to training-ready features.

## Stages

```
Raw Data
  |
  v
Phase 1: EDA (Explore)
  - Profile columns, check distributions, flag quality issues
  - Output: EDA report with findings and recommendations
  |
  v
Phase 2: Clean
  - Deduplicate, fix types, handle missing, remove outliers, validate
  - Output: clean.csv
  |
  v
Phase 3: Engineer
  - Create features, encode categories, add interactions, rolling windows
  - Output: features.csv
  |
  v
Ready for Training
```

## Quick reference

| Phase | Script | Input | Output |
|-------|--------|-------|--------|
| EDA | `eda.py` | raw data | report (stderr) |
| Clean | `clean.py` | raw data | clean.csv |
| Engineer | `engineer_features.py` | clean.csv | features.csv |

Each phase feeds the next. Never skip EDA -- findings drive cleaning. Never engineer from raw data -- clean first.
