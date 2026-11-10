package main

import (
	"fmt"
	"math"
)

// ==========================================
// NGDS ASSIGNMENT 2
// PHASE 4: LINEAR REGRESSION
// ==========================================

// Structure for linear regression coefficients
type LinearModel struct {
	Intercept float64
	Slope     float64
}

// Fit a linear regression model using least squares.
// Degenerate inputs return a flat model with zero slope to avoid NaN/Inf values.
func fitLinear(data []Observation) LinearModel {

	if len(data) == 0 {
		return LinearModel{Intercept: 0, Slope: 0}
	}

	if len(data) == 1 {
		return LinearModel{
			Intercept: data[0].Emissions,
			Slope:     0,
		}
	}

	n := float64(len(data))

	sumX := 0.0
	sumY := 0.0

	// Centre years around 1970
	for _, obs := range data {
		sumX += float64(obs.Year - 1970)
		sumY += obs.Emissions
	}

	meanX := sumX / n
	meanY := sumY / n

	numerator := 0.0
	denominator := 0.0

	for _, obs := range data {

		x := float64(obs.Year-1970) - meanX
		y := obs.Emissions - meanY

		numerator += x * y
		denominator += x * x
	}

	if math.Abs(denominator) < 1e-12 {
		return LinearModel{Intercept: meanY, Slope: 0}
	}

	slope := numerator / denominator
	intercept := meanY - slope*meanX

	return LinearModel{
		Intercept: intercept,
		Slope:     slope,
	}
}

// Predict CO2 emissions for a given year
func (m LinearModel) Predict(year int) float64 {

	x := float64(year - 1970)

	return m.Intercept + m.Slope*x
}

// Evaluate the model using test observations
func evaluateLinear(
	model LinearModel,
	test []Observation,
) {

	if len(test) == 0 {
		fmt.Println("\nTEST SET PREDICTIONS")
		fmt.Println("----------------------------------------")
		fmt.Println("No test observations available.")
		fmt.Println("\nLINEAR REGRESSION RESULTS")
		fmt.Println("----------------------------------------")
		fmt.Println("MAE: undefined")
		fmt.Println("RMSE: undefined")
		fmt.Println("Test R2: undefined")
		return
	}

	n := float64(len(test))

	mae := 0.0
	mse := 0.0
	meanActual := 0.0

	for _, obs := range test {
		meanActual += obs.Emissions
	}

	meanActual /= n

	ssRes := 0.0
	ssTot := 0.0

	fmt.Println("\nTEST SET PREDICTIONS")
	fmt.Println("----------------------------------------")
	fmt.Println("Year | Actual | Predicted")

	for _, obs := range test {

		predicted := model.Predict(obs.Year)
		errorValue := obs.Emissions - predicted

		mae += math.Abs(errorValue)
		mse += errorValue * errorValue

		ssRes += errorValue * errorValue

		ssTot += math.Pow(
			obs.Emissions-meanActual, 2,
		)

		fmt.Printf(
			"%d | %.4f | %.4f\n",
			obs.Year,
			obs.Emissions,
			predicted,
		)
	}

	mae /= n
	rmse := math.Sqrt(mse / n)

	fmt.Println("\nLINEAR REGRESSION RESULTS")
	fmt.Println("----------------------------------------")

	fmt.Printf("MAE: %.4f MtCO2e\n", mae)
	fmt.Printf("RMSE: %.4f MtCO2e\n", rmse)

	if ssTot > 0 {
		r2 := 1 - ssRes/ssTot
		fmt.Printf("Test R2: %.4f\n", r2)
	} else {
		fmt.Println("Test R2: undefined")
	}
}

// Main regression experiment
func runRegression(data []Observation) {

	fmt.Println("\nPHASE 4: LINEAR REGRESSION")
	fmt.Println("----------------------------------------")

	if len(data) < 2 {
		fmt.Println("Insufficient data for regression.")
		return
	}

	// Chronological 80/20 split
	split := int(float64(len(data)) * 0.8)

	if split <= 0 || split >= len(data) {
		fmt.Println("Insufficient data for regression split.")
		return
	}

	train := data[:split]
	test := data[split:]

	fmt.Printf(
		"Training: %d-%d (%d observations)\n",
		train[0].Year,
		train[len(train)-1].Year,
		len(train),
	)

	fmt.Printf(
		"Testing: %d-%d (%d observations)\n",
		test[0].Year,
		test[len(test)-1].Year,
		len(test),
	)

	// Train linear regression model
	model := fitLinear(train)

	fmt.Printf(
		"\nIntercept: %.4f\nSlope: %.4f MtCO2e/year\n",
		model.Intercept,
		model.Slope,
	)

	// Evaluate predictions
	evaluateLinear(model, test)
}
