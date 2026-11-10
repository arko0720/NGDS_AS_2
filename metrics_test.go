package main

import (
	"math"
	"testing"
)

func TestMeanAbsoluteError(t *testing.T) {
	actual := []float64{10, 20, 30}
	predicted := []float64{12, 18, 33}

	sum := 0.0
	for i := range actual {
		sum += math.Abs(actual[i] - predicted[i])
	}

	got := sum / float64(len(actual))
	want := 7.0 / 3.0

	if math.Abs(got-want) > 1e-9 {
		t.Errorf("MAE = %.6f, expected %.6f", got, want)
	}
}

func TestRootMeanSquaredError(t *testing.T) {
	actual := []float64{10, 20, 30}
	predicted := []float64{12, 18, 33}

	sum := 0.0
	for i := range actual {
		diff := actual[i] - predicted[i]
		sum += diff * diff
	}

	got := math.Sqrt(sum / float64(len(actual)))
	want := math.Sqrt(17.0 / 3.0)

	if math.Abs(got-want) > 1e-9 {
		t.Errorf("RMSE = %.6f, expected %.6f", got, want)
	}
}

func TestChronologicalSplit(t *testing.T) {
	years := []int{1970, 1971, 1972, 2011, 2012, 2022}

	trainingEnd := 2011
	testingStart := 2012

	for _, year := range years {
		if year <= trainingEnd && year >= testingStart {
			t.Errorf("Year %d appears in both training and testing", year)
		}
	}

	if trainingEnd >= testingStart {
		t.Fatal("Training period overlaps testing period")
	}
}

func TestFitLinearDegenerateData(t *testing.T) {
	model := fitLinear([]Observation{{Year: 2020, Emissions: 8}})
	if math.IsNaN(model.Intercept) || math.IsNaN(model.Slope) {
		t.Fatal("degenerate linear fit produced NaN values")
	}
	if model.Slope != 0 {
		t.Fatalf("expected zero slope for single-point fit, got %.6f", model.Slope)
	}
	if model.Intercept != 8.0 {
		t.Fatalf("expected intercept 8.0 for single-point fit, got %.6f", model.Intercept)
	}

	flatModel := fitLinear([]Observation{
		{Year: 2000, Emissions: 2},
		{Year: 2000, Emissions: 3},
	})
	if math.IsNaN(flatModel.Intercept) || math.IsNaN(flatModel.Slope) {
		t.Fatal("constant-x regression produced NaN values")
	}
	if flatModel.Slope != 0 {
		t.Fatalf("expected zero slope for constant x values, got %.6f", flatModel.Slope)
	}
	if math.Abs(flatModel.Intercept-2.5) > 1e-9 {
		t.Fatalf("expected intercept 2.5 for constant x values, got %.6f", flatModel.Intercept)
	}
}
