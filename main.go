package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

// ==================================================
// NGDS ASSIGNMENT 2
// GERMANY CO2 EMISSIONS ANALYSIS
//
// PHASE 1: DATASET LOADING
// PHASE 2: DATA CLEANING AND STATISTICS
// PHASE 3: EXPLORATORY DATA VISUALISATION
// ==================================================

// Observation represents one year of CO2 emissions.
type Observation struct {
	Year      int
	Emissions float64
}

func main() {

	fmt.Println("========================================")
	fmt.Println("NGDS ASSIGNMENT 2")
	fmt.Println("GERMANY CO2 EMISSIONS ANALYSIS")
	fmt.Println("========================================")

	// ==========================================
	// PHASE 1: DATASET LOADING
	// ==========================================

	fmt.Println("\nPHASE 1: DATASET LOADING")

	file, err := os.Open("NGDS_CO2_Germany.csv")

	if err != nil {
		fmt.Println("Error opening dataset:", err)
		return
	}

	defer file.Close()

	reader := csv.NewReader(file)

	records, err := reader.ReadAll()

	if err != nil {
		fmt.Println("Error reading CSV:", err)
		return
	}

	if len(records) < 2 {
		fmt.Println("Error: Dataset has no observations.")
		return
	}

	fmt.Println("Dataset loaded successfully.")
	fmt.Println("Number of observations:", len(records)-1)
	fmt.Println("Number of columns:", len(records[0]))

	// ==========================================
	// PHASE 2: DATA CLEANING
	// ==========================================

	fmt.Println("\nPHASE 2: DATA CLEANING")

	// STEP 1: Validate expected column headers
	expected := []string{
		"Country Name",
		"Country Code",
		"Indicator Name",
		"Indicator Code",
		"Year",
		"CO2_Mt_CO2e",
	}

	if len(records[0]) != len(expected) {
		fmt.Println("Error: Unexpected number of columns.")
		return
	}

	for i, name := range expected {
		if strings.TrimSpace(records[0][i]) != name {
			fmt.Println("Error: Unexpected column:", records[0][i])
			return
		}
	}

	fmt.Println("Column headers validated successfully.")

	// STEP 2: Clean and validate observations
	var data []Observation

	missing := 0
	invalid := 0
	duplicates := 0

	seenYears := make(map[int]bool)

	for _, row := range records[1:] {

		if len(row) != len(expected) {
			invalid++
			continue
		}

		yearText := strings.TrimSpace(row[4])
		emissionText := strings.TrimSpace(row[5])

		// Check missing values
		if yearText == "" || emissionText == "" {
			missing++
			continue
		}

		// Convert year to integer
		year, err1 := strconv.Atoi(yearText)

		// Convert emissions to float64
		emission, err2 := strconv.ParseFloat(
			emissionText, 64,
		)

		// Check invalid numerical values
		if err1 != nil ||
			err2 != nil ||
			math.IsNaN(emission) ||
			math.IsInf(emission, 0) ||
			emission < 0 {

			invalid++
			continue
		}

		// Check duplicate years
		if seenYears[year] {
			duplicates++
			continue
		}

		seenYears[year] = true

		data = append(data, Observation{
			Year:      year,
			Emissions: emission,
		})
	}

	if len(data) == 0 {
		fmt.Println("Error: No valid observations.")
		return
	}

	// STEP 3: Sort data chronologically
	sort.Slice(data, func(i, j int) bool {
		return data[i].Year < data[j].Year
	})

	// STEP 4: Display data quality report
	fmt.Println("\nDATA QUALITY REPORT")
	fmt.Println("----------------------------------------")

	fmt.Println("Original observations:", len(records)-1)
	fmt.Println("Valid observations:", len(data))
	fmt.Println("Missing observations:", missing)
	fmt.Println("Invalid observations:", invalid)
	fmt.Println("Duplicate years:", duplicates)

	// ==========================================
	// PHASE 2: DESCRIPTIVE STATISTICS
	// ==========================================

	fmt.Println("\nDESCRIPTIVE STATISTICS")
	fmt.Println("----------------------------------------")

	n := len(data)
	sum := 0.0

	minObs := data[0]
	maxObs := data[0]

	values := make([]float64, 0, n)

	// Calculate sum, minimum and maximum
	for _, obs := range data {

		sum += obs.Emissions

		values = append(values, obs.Emissions)

		if obs.Emissions < minObs.Emissions {
			minObs = obs
		}

		if obs.Emissions > maxObs.Emissions {
			maxObs = obs
		}
	}

	// Calculate mean
	mean := sum / float64(n)

	// Calculate sample standard deviation
	varianceSum := 0.0

	for _, value := range values {
		difference := value - mean
		varianceSum += difference * difference
	}

	stdDev := 0.0

	if n > 1 {
		variance := varianceSum / float64(n-1)
		stdDev = math.Sqrt(variance)
	}

	// Calculate median
	sort.Float64s(values)

	median := 0.0

	if n%2 == 0 {
		median = (values[n/2-1] + values[n/2]) / 2
	} else {
		median = values[n/2]
	}

	// Display descriptive statistics
	fmt.Printf("Mean CO2 emissions: %.4f MtCO2e\n", mean)

	fmt.Printf("Median CO2 emissions: %.4f MtCO2e\n", median)

	fmt.Printf("Standard deviation: %.4f MtCO2e\n", stdDev)

	fmt.Printf(
		"Minimum emissions: %.4f MtCO2e (Year %d)\n",
		minObs.Emissions,
		minObs.Year,
	)

	fmt.Printf(
		"Maximum emissions: %.4f MtCO2e (Year %d)\n",
		maxObs.Emissions,
		maxObs.Year,
	)

	fmt.Printf(
		"Year range: %d - %d\n",
		data[0].Year,
		data[n-1].Year,
	)

	// ==========================================
	// PHASE 3: EXPLORATORY DATA VISUALISATION
	// ==========================================

	fmt.Println("\nPHASE 3: DATA VISUALISATION")
	fmt.Println("----------------------------------------")

	// Call plotting function from eda.go
	err = plotEmissions(data)

	if err != nil {
		fmt.Println("Error generating CO2 graph:", err)
		return
	}

	// ==========================================
	// PHASE 4: LINEAR REGRESSION
	// ==========================================

	runRegression(data)

	// ==========================================
	// PHASE 5: NAIVE BASELINE MODEL
	// ==========================================

	runBaseline(data)

	// ==========================================
	// PHASE 6: QUADRATIC REGRESSION
	// ==========================================

	runQuadratic(data)

	// ==========================================
	// PHASE 7: ROLLING-ORIGIN VALIDATION
	// ==========================================

	runValidation(data)
	// ==========================================
	// PHASE 8: FINAL VISUALISATIONS AND EXPORT
	// ==========================================

	if err := runResults(data); err != nil {
		fmt.Println("Phase 8 error:", err)
		return
	}
	// ==========================================
	// PROGRAM COMPLETED
	// ==========================================

	fmt.Println("\n========================================")
	fmt.Println("PHASES 1 TO 8 COMPLETED SUCCESSFULLY")
	fmt.Println("========================================")
}
