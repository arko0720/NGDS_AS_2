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
