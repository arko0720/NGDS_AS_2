package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// ==========================================
// PHASE 6: QUADRATIC REGRESSION
// ==========================================

type QuadraticModel struct {
	B0 float64
	B1 float64
	B2 float64
}

// Fit a quadratic model using QR decomposition.
func fitQuadratic(data []Observation) (QuadraticModel, error) {

	n := len(data)

	if n < 3 {
		return QuadraticModel{}, fmt.Errorf(
			"at least 3 observations required",
		)
	}

	X := mat.NewDense(n, 3, nil)
	Y := mat.NewDense(n, 1, nil)

	for i, obs := range data {
		x := float64(obs.Year - 1970)

		X.Set(i, 0, 1)
		X.Set(i, 1, x)
		X.Set(i, 2, x*x)

		Y.Set(i, 0, obs.Emissions)
	}

	var qr mat.QR
	qr.Factorize(X)

	var coefficients mat.Dense

	if err := qr.SolveTo(&coefficients, false, Y); err != nil {
		return QuadraticModel{}, err
	}

	return QuadraticModel{
		B0: coefficients.At(0, 0),
		B1: coefficients.At(1, 0),
		B2: coefficients.At(2, 0),
	}, nil
}

// Predict emissions for a given year.
func (m QuadraticModel) Predict(year int) float64 {
	x := float64(year - 1970)
	return m.B0 + m.B1*x + m.B2*x*x
}

// Evaluate the quadratic model on unseen data.
func evaluateQuadratic(model QuadraticModel, test []Observation) {

	n := float64(len(test))

	meanActual := 0.0
	for _, obs := range test {
		meanActual += obs.Emissions
	}
	meanActual /= n

	sumAbsolute := 0.0
	sumSquared := 0.0
	ssTotal := 0.0

	fmt.Println("\nQUADRATIC TEST PREDICTIONS")
	fmt.Println("----------------------------------------")
	fmt.Println("Year | Actual | Predicted")

	for _, obs := range test {

		predicted := model.Predict(obs.Year)
		err := obs.Emissions - predicted

		sumAbsolute += math.Abs(err)
		sumSquared += err * err

		diff := obs.Emissions - meanActual
		ssTotal += diff * diff

		fmt.Printf(
			"%d | %.4f | %.4f\n",
			obs.Year, obs.Emissions, predicted,
		)
	}

	mae := sumAbsolute / n
	rmse := math.Sqrt(sumSquared / n)

	fmt.Println("\nQUADRATIC REGRESSION RESULTS")
	fmt.Println("----------------------------------------")
	fmt.Printf("Test MAE: %.4f MtCO2e\n", mae)
	fmt.Printf("Test RMSE: %.4f MtCO2e\n", rmse)

	if ssTotal > 0 {
		fmt.Printf("Test R2: %.4f\n",
			1-sumSquared/ssTotal)
	} else {
		fmt.Println("Test R2: undefined")
	}
}

// Run the quadratic regression experiment.
func runQuadratic(data []Observation) {

	fmt.Println("\nPHASE 6: QUADRATIC REGRESSION")
	fmt.Println("----------------------------------------")

	split := int(float64(len(data)) * 0.8)

	if split < 3 || split >= len(data) {
		fmt.Println("Insufficient data for evaluation.")
		return
	}

	train := data[:split]
	test := data[split:]

	model, err := fitQuadratic(train)

	if err != nil {
		fmt.Println("Regression error:", err)
		return
	}

	fmt.Printf("B0: %.6f\n", model.B0)
	fmt.Printf("B1: %.6f\n", model.B1)
	fmt.Printf("B2: %.6f\n", model.B2)

	evaluateQuadratic(model, test)
}
