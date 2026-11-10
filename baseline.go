package main

import (
	"fmt"
	"math"
)

// PHASE 5: NAIVE BASELINE MODEL
// Predict the last training observation for every test year.

func runBaseline(data []Observation) {

	split := int(float64(len(data)) * 0.8)

	if split < 1 || split >= len(data) {
		fmt.Println("Insufficient data for baseline evaluation.")
		return
	}

	train := data[:split]
	test := data[split:]

	lastValue := train[len(train)-1].Emissions

	sumAbsError := 0.0
	sumSquaredError := 0.0
	sumActual := 0.0

	for _, obs := range test {
		sumActual += obs.Emissions
	}

	meanActual := sumActual / float64(len(test))
	ssTotal := 0.0

	for _, obs := range test {
		err := obs.Emissions - lastValue

		sumAbsError += math.Abs(err)
		sumSquaredError += err * err

		diff := obs.Emissions - meanActual
		ssTotal += diff * diff
	}

	n := float64(len(test))

	mae := sumAbsError / n
	rmse := math.Sqrt(sumSquaredError / n)

	fmt.Println("\nPHASE 5: NAIVE BASELINE")
	fmt.Println("----------------------------------------")
	fmt.Printf("Last training year: %d\n",
		train[len(train)-1].Year)
	fmt.Printf("Constant prediction: %.4f MtCO2e\n",
		lastValue)
	fmt.Printf("Test MAE: %.4f MtCO2e\n", mae)
	fmt.Printf("Test RMSE: %.4f MtCO2e\n", rmse)

	if ssTotal > 0 {
		r2 := 1 - sumSquaredError/ssTotal
		fmt.Printf("Test R2: %.4f\n", r2)
	} else {
		fmt.Println("Test R2: undefined")
	}
}
