package main

import (
	"math"
	"testing"
)

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

// Test 1: Verify OLS coefficients against a known line.
func TestLinearKnownCoefficients(t *testing.T) {
	data := []Observation{
		{Year: 1970, Emissions: 100},
		{Year: 1971, Emissions: 90},
		{Year: 1972, Emissions: 80},
		{Year: 1973, Emissions: 70},
	}

	model := fitLinear(data)

	if !almostEqual(model.Intercept, 100) {
		t.Errorf("intercept = %v, want 100", model.Intercept)
	}
	if !almostEqual(model.Slope, -10) {
		t.Errorf("slope = %v, want -10", model.Slope)
	}
}

// Test 2: Verify predictions from a known model.
func TestLinearKnownPredictions(t *testing.T) {
	model := LinearModel{
		Intercept: 100,
		Slope:     -10,
	}

	cases := []struct {
		year int
		want float64
	}{
		{1970, 100},
		{1975, 50},
		{1980, 0},
	}

	for _, tc := range cases {
		got := model.Predict(tc.year)
		if !almostEqual(got, tc.want) {
			t.Errorf("year %d: got %v, want %v",
				tc.year, got, tc.want)
		}
	}
}

// Test 3: Verify fitting does not modify observations.
func TestLinearDoesNotModifyInput(t *testing.T) {
	data := []Observation{
		{Year: 1970, Emissions: 100},
		{Year: 1971, Emissions: 90},
		{Year: 1972, Emissions: 80},
	}

	original := append([]Observation(nil), data...)
	_ = fitLinear(data)

	for i := range data {
		if data[i].Year != original[i].Year ||
			data[i].Emissions != original[i].Emissions {
			t.Fatalf("observation %d was modified", i)
		}
	}
}

// Test 4: Verify constant target produces a flat line.
func TestLinearConstantEmissions(t *testing.T) {
	data := []Observation{
		{Year: 1970, Emissions: 500},
		{Year: 1971, Emissions: 500},
		{Year: 1972, Emissions: 500},
	}

	model := fitLinear(data)

	if !almostEqual(model.Slope, 0) ||
		!almostEqual(model.Intercept, 500) {
		t.Fatalf("unexpected model: %+v", model)
	}
}

// Test 5: Verify empty input is handled consistently.
func TestLinearEmptyInput(t *testing.T) {
	model := fitLinear(nil)

	if model.Intercept != 0 || model.Slope != 0 {
		t.Fatalf("unexpected empty-input model: %+v", model)
	}
}
