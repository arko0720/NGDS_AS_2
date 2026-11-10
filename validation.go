package main

import (
	"fmt"
	"math"
)

// ==========================================
// PHASE 7: ROLLING-ORIGIN CROSS-VALIDATION
// ==========================================

// Calculate MAE and RMSE from prediction errors.
func validationMetrics(errors []float64) (float64, float64) {

	if len(errors) == 0 {
		return math.NaN(), math.NaN()
	}

	sumAbsolute := 0.0
	sumSquared := 0.0

	for _, err := range errors {
		sumAbsolute += math.Abs(err)
		sumSquared += err * err
	}

	n := float64(len(errors))

	mae := sumAbsolute / n
	rmse := math.Sqrt(sumSquared / n)

	return mae, rmse
}

// Run expanding-window one-year-ahead validation.
func runValidation(data []Observation) {

	fmt.Println("\nPHASE 7: ROLLING-ORIGIN VALIDATION")
	fmt.Println("----------------------------------------")

	// Keep the final 20% completely outside validation.
	split := int(float64(len(data)) * 0.8)

	trainingData := data[:split]

	// Begin validation after 20 years of training.
	initialWindow := 20

	if len(trainingData) <= initialWindow {
		fmt.Println("Insufficient data for validation.")
		return
	}

	var baselineErrors []float64
	var linearErrors []float64
	var quadraticErrors []float64

	fmt.Println("Expanding-window, one-year-ahead forecasts")
	fmt.Println("Year | Actual | Baseline | Linear | Quadratic")

	for i := initialWindow; i < len(trainingData); i++ {

		// Only historical observations are used.
		train := trainingData[:i]

		// The next observation is the validation target.
		target := trainingData[i]

		// Model 1: Naive baseline
		baselinePrediction := train[len(train)-1].Emissions

		// Model 2: Linear regression
		linearModel := fitLinear(train)
		linearPrediction := linearModel.Predict(target.Year)

		// Model 3: Quadratic regression
		quadraticModel, err := fitQuadratic(train)

		if err != nil {
			fmt.Println("Quadratic fitting error:", err)
			return
		}

		quadraticPrediction := quadraticModel.Predict(target.Year)

		// Store prediction errors.
		baselineErrors = append(
			baselineErrors,
			target.Emissions-baselinePrediction,
		)

		linearErrors = append(
			linearErrors,
			target.Emissions-linearPrediction,
		)

		quadraticErrors = append(
			quadraticErrors,
			target.Emissions-quadraticPrediction,
		)

		fmt.Printf(
			"%d | %.2f | %.2f | %.2f | %.2f\n",
			target.Year,
			target.Emissions,
			baselinePrediction,
			linearPrediction,
			quadraticPrediction,
		)
	}

	// Aggregate validation metrics.
	baselineMAE, baselineRMSE := validationMetrics(baselineErrors)
	linearMAE, linearRMSE := validationMetrics(linearErrors)
	quadraticMAE, quadraticRMSE := validationMetrics(quadraticErrors)

	fmt.Println("\nROLLING-ORIGIN VALIDATION RESULTS")
	fmt.Println("----------------------------------------")

	fmt.Printf("Validation years: %d-%d\n",
		trainingData[initialWindow].Year,
		trainingData[len(trainingData)-1].Year,
	)

	fmt.Printf("Number of folds: %d\n", len(linearErrors))

	fmt.Println("\nModel       | MAE       | RMSE")
	fmt.Println("----------------------------------------")

	fmt.Printf("Baseline    | %9.4f | %9.4f\n",
		baselineMAE, baselineRMSE)

	fmt.Printf("Linear      | %9.4f | %9.4f\n",
		linearMAE, linearRMSE)

	fmt.Printf("Quadratic   | %9.4f | %9.4f\n",
		quadraticMAE, quadraticRMSE)
}
