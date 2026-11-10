# NGDS Assignment 2 — Germany CO₂ Emissions Forecasting

## Project Overview

This project implements a Go-based analytical and forecasting pipeline for Germany's annual CO₂ emissions from 1970 to 2022.

## Dataset

- Source: Lecturer-provided `NGDS_CO2_Germany.csv`
- Observations: 53
- Period: 1970–2022
- Target: CO₂ emissions (MtCO₂e)

## Methodology

The project implements eight phases covering data loading, cleaning, exploratory data analysis, linear regression, naive baseline forecasting, quadratic regression, rolling-origin validation, and results export.

## Model Performance

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Linear Regression | 31.2213 | 35.2197 | 0.6572 |
| Naive Baseline | 54.1380 | 75.9480 | -0.5942 |
| Quadratic Regression | 74.1423 | 77.7196 | -0.6695 |

The test period covers 2012–2022. Linear regression achieves the lowest error on this fixed-origin holdout.

Rolling-origin one-year-ahead validation produces a different ranking, with the naive baseline achieving the lowest MAE.

## Running the Project

Requires Go and the dependencies specified in `go.mod`.

```bash
go mod download
go run .
```

## Testing

```bash
go test -count=1 -v ./...
go vet ./...
```

## Outputs

- `germany_co2_trend.png`
- `model_predictions.png`
- `model_error_comparison.png`
- `test_predictions.csv`
- `model_metrics.csv`

## Generative AI

ChatGPT supported implementation and analysis. GitHub Copilot was used for independent code review and improvements. Changes were checked using Go tests, static analysis and project execution.

Supporting prompt and execution evidence is documented separately in the GenAI evidence report.
