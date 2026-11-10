package main

import (
	"math"
	"testing"
)

func TestValidationMetricsKnownValues(t *testing.T) {
	errors := []float64{2, -2, 3}

	mae, rmse := validationMetrics(errors)

	wantMAE := 7.0 / 3.0
	wantRMSE := math.Sqrt(17.0 / 3.0)

	if math.Abs(mae-wantMAE) > 1e-9 {
		t.Errorf("MAE = %.8f, want %.8f", mae, wantMAE)
	}

	if math.Abs(rmse-wantRMSE) > 1e-9 {
		t.Errorf("RMSE = %.8f, want %.8f", rmse, wantRMSE)
	}
}

func TestValidationMetricsZeroErrors(t *testing.T) {
	mae, rmse := validationMetrics(
		[]float64{0, 0, 0},
	)

	if mae != 0 || rmse != 0 {
		t.Fatalf(
			"expected zero errors, got MAE=%v RMSE=%v",
			mae, rmse,
		)
	}
}

func TestValidationMetricsEmpty(t *testing.T) {
	mae, rmse := validationMetrics(nil)

	if !math.IsNaN(mae) || !math.IsNaN(rmse) {
		t.Fatalf(
			"expected NaN metrics for empty input",
		)
	}
}

func TestRollingOriginWindowBoundaries(t *testing.T) {
	data := make([]Observation, 53)

	for i := range data {
		data[i] = Observation{
			Year:      1970 + i,
			Emissions: float64(1000 - i*5),
		}
	}

	split := int(float64(len(data)) * 0.8)
	trainingData := data[:split]
	initialWindow := 20

	folds := 0

	for i := initialWindow; i < len(trainingData); i++ {
		train := trainingData[:i]
		target := trainingData[i]

		if train[len(train)-1].Year >= target.Year {
			t.Fatalf(
				"future-data leakage at year %d",
				target.Year,
			)
		}

		if target.Year >= 2012 {
			t.Fatalf(
				"holdout year %d entered validation",
				target.Year,
			)
		}

		if target.Year-train[len(train)-1].Year != 1 {
			t.Fatalf(
				"forecast horizon is not one year",
			)
		}

		folds++
	}

	if folds != 22 {
		t.Fatalf("got %d folds, want 22", folds)
	}
}
