package main

import (
	"encoding/csv"
	"fmt"
	"image/color"
	"math"
	"os"
	"strconv"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// ==================================================
// NGDS ASSIGNMENT 2
// PHASE 8: FINAL VISUALISATIONS AND RESULTS EXPORT
// ==================================================

// ModelResult stores the evaluation metrics.
type ModelResult struct {
	Name string
	MAE  float64
	RMSE float64
	R2   float64
}

// ==================================================
// STEP 1: CALCULATE MODEL EVALUATION METRICS
// ==================================================

func calculateMetrics(actual, predicted []float64) ModelResult {

	n := len(actual)

	if n == 0 || n != len(predicted) {
		return ModelResult{
			MAE:  math.NaN(),
			RMSE: math.NaN(),
			R2:   math.NaN(),
		}
	}

	mean := 0.0

	for _, value := range actual {
		mean += value
	}

	mean /= float64(n)

	absoluteSum := 0.0
	squaredSum := 0.0
	totalSum := 0.0

	for i := range actual {

		err := actual[i] - predicted[i]

		absoluteSum += math.Abs(err)
		squaredSum += err * err

		diff := actual[i] - mean
		totalSum += diff * diff
	}

	r2 := math.NaN()

	if totalSum > 0 {
		r2 = 1 - squaredSum/totalSum
	}

	return ModelResult{
		MAE:  absoluteSum / float64(n),
		RMSE: math.Sqrt(squaredSum / float64(n)),
		R2:   r2,
	}
}

// ==================================================
// STEP 2: EXPORT RESULTS TO CSV
// ==================================================

func saveCSV(filename string, rows [][]string) error {

	file, err := os.Create(filename)

	if err != nil {
		return err
	}

	defer file.Close()

	writer := csv.NewWriter(file)

	if err := writer.WriteAll(rows); err != nil {
		return err
	}

	return writer.Error()
}

// ==================================================
// STEP 3: CREATE COLOURED PREDICTION LINE GRAPH
// ==================================================

func makeLinePlot(
	filename string,
	title string,
	series []plotter.XYs,
	names []string,
) error {

	if len(series) != len(names) {
		return fmt.Errorf("series and legend names must match")
	}

	p := plot.New()

	p.Title.Text = title
	p.X.Label.Text = "Year"
	p.Y.Label.Text = "CO2 emissions (MtCO2e)"

	p.Add(plotter.NewGrid())

	// Distinct colours for the four series.
	colors := []color.Color{
		color.RGBA{30, 70, 130, 255},   // Actual: blue
		color.RGBA{130, 130, 130, 255}, // Baseline: grey
		color.RGBA{20, 140, 90, 255},   // Linear: green
		color.RGBA{210, 90, 50, 255},   // Quadratic: orange
	}

	for i, points := range series {

		line, err := plotter.NewLine(points)

		if err != nil {
			return err
		}

		line.Color = colors[i%len(colors)]
		line.Width = vg.Points(2)

		// Dashed line for the naive baseline.
		if i == 1 {
			line.Dashes = []vg.Length{
				vg.Points(6),
				vg.Points(4),
			}
		}

		p.Add(line)
		p.Legend.Add(names[i], line)
	}

	p.Legend.Top = true
	p.Legend.Left = true

	// Position the legend above the plot.
	p.Legend.YOffs = vg.Points(10)

	return p.Save(
		10*vg.Inch,
		6*vg.Inch,
		filename,
	)
}

// ==================================================
// STEP 4: CREATE GROUPED BAR CHART
// ==================================================

func makeErrorComparisonPlot(
	filename string,
	results []ModelResult,
) error {

	p := plot.New()

	p.Title.Text = "Model Error Comparison (2012-2022)"
	p.Y.Label.Text = "Prediction error (MtCO2e)"

	p.Add(plotter.NewGrid())

	maeValues := make(plotter.Values, len(results))
	rmseValues := make(plotter.Values, len(results))

	names := make([]string, len(results))

	for i, result := range results {
		maeValues[i] = result.MAE
		rmseValues[i] = result.RMSE
		names[i] = result.Name
	}

	barWidth := vg.Points(25)

	// MAE bars
	maeBars, err := plotter.NewBarChart(
		maeValues,
		barWidth,
	)

	if err != nil {
		return err
	}

	maeBars.Color = color.RGBA{
		40, 110, 180, 255,
	}

	maeBars.Offset = -barWidth / 2

	// RMSE bars
	rmseBars, err := plotter.NewBarChart(
		rmseValues,
		barWidth,
	)

	if err != nil {
		return err
	}

	rmseBars.Color = color.RGBA{
		220, 125, 45, 255,
	}

	rmseBars.Offset = barWidth / 2

	p.Add(maeBars, rmseBars)

	p.Legend.Add("MAE", maeBars)
	p.Legend.Add("RMSE", rmseBars)

	p.Legend.Top = true
	p.Legend.Left = true

	p.NominalX(names...)

	p.Y.Min = 0

	return p.Save(
		10*vg.Inch,
		6*vg.Inch,
		filename,
	)
}

// ==================================================
// STEP 5: EXECUTE PHASE 8
// ==================================================

func runResults(data []Observation) error {

	fmt.Println("\nPHASE 8: VISUALISATIONS AND EXPORT")
	fmt.Println("----------------------------------------")

	// Chronological 80/20 split.
	split := int(float64(len(data)) * 0.8)

	if split < 3 || split >= len(data) {
		return fmt.Errorf("insufficient observations")
	}

	train := data[:split]
	test := data[split:]

	// Fit linear regression using training data.
	linear := fitLinear(train)

	// Fit quadratic regression using training data.
	quadratic, err := fitQuadratic(train)

	if err != nil {
		return err
	}

	// Naive baseline: final training observation.
	baselineValue := train[len(train)-1].Emissions

	// Store actual values and predictions.
	actual := make([]float64, 0, len(test))
	baselinePred := make([]float64, 0, len(test))
	linearPred := make([]float64, 0, len(test))
	quadraticPred := make([]float64, 0, len(test))

	// Plotting coordinates.
	actualPoints := make(plotter.XYs, 0, len(test))
	baselinePoints := make(plotter.XYs, 0, len(test))
	linearPoints := make(plotter.XYs, 0, len(test))
	quadraticPoints := make(plotter.XYs, 0, len(test))

	// CSV header.
	rows := [][]string{{
		"Year",
		"Actual",
		"Baseline",
		"Linear",
		"Quadratic",
	}}

	// ==========================================
	// GENERATE TEST SET PREDICTIONS
	// ==========================================

	for _, obs := range test {

		year := float64(obs.Year)

		a := obs.Emissions
		b := baselineValue
		l := linear.Predict(obs.Year)
		q := quadratic.Predict(obs.Year)

		actual = append(actual, a)
		baselinePred = append(baselinePred, b)
		linearPred = append(linearPred, l)
		quadraticPred = append(quadraticPred, q)

		actualPoints = append(
			actualPoints,
			plotter.XY{X: year, Y: a},
		)

		baselinePoints = append(
			baselinePoints,
			plotter.XY{X: year, Y: b},
		)

		linearPoints = append(
			linearPoints,
			plotter.XY{X: year, Y: l},
		)

		quadraticPoints = append(
			quadraticPoints,
			plotter.XY{X: year, Y: q},
		)

		rows = append(rows, []string{
			strconv.Itoa(obs.Year),
			strconv.FormatFloat(a, 'f', 6, 64),
			strconv.FormatFloat(b, 'f', 6, 64),
			strconv.FormatFloat(l, 'f', 6, 64),
			strconv.FormatFloat(q, 'f', 6, 64),
		})
	}

	// ==========================================
	// EXPORT TEST PREDICTIONS
	// ==========================================

	if err := saveCSV(
		"test_predictions.csv",
		rows,
	); err != nil {
		return err
	}

	// ==========================================
	// GENERATE ACTUAL VS PREDICTED GRAPH
	// ==========================================

	if err := makeLinePlot(
		"model_predictions.png",
		"Germany CO2: Actual vs Predicted (2012-2022)",
		[]plotter.XYs{
			actualPoints,
			baselinePoints,
			linearPoints,
			quadraticPoints,
		},
		[]string{
			"Actual",
			"Baseline",
			"Linear",
			"Quadratic",
		},
	); err != nil {
		return err
	}

	// ==========================================
	// CALCULATE FINAL MODEL METRICS
	// ==========================================

	results := []ModelResult{
		calculateMetrics(actual, baselinePred),
		calculateMetrics(actual, linearPred),
		calculateMetrics(actual, quadraticPred),
	}

	results[0].Name = "Baseline"
	results[1].Name = "Linear"
	results[2].Name = "Quadratic"

	// ==========================================
	// EXPORT MODEL EVALUATION METRICS
	// ==========================================

	metricRows := [][]string{{
		"Model",
		"MAE",
		"RMSE",
		"R2",
	}}

	for _, result := range results {

		metricRows = append(metricRows, []string{
			result.Name,
			strconv.FormatFloat(result.MAE, 'f', 6, 64),
			strconv.FormatFloat(result.RMSE, 'f', 6, 64),
			strconv.FormatFloat(result.R2, 'f', 6, 64),
		})
	}

	if err := saveCSV(
		"model_metrics.csv",
		metricRows,
	); err != nil {
		return err
	}

	// ==========================================
	// GENERATE GROUPED BAR CHART
	// ==========================================

	if err := makeErrorComparisonPlot(
		"model_error_comparison.png",
		results,
	); err != nil {
		return err
	}

	// ==========================================
	// COMPLETION MESSAGE
	// ==========================================

	fmt.Println("Saved: test_predictions.csv")
	fmt.Println("Saved: model_metrics.csv")
	fmt.Println("Saved: model_predictions.png")
	fmt.Println("Saved: model_error_comparison.png")

	return nil
}
