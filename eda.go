package main

import (
	"fmt"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// PHASE 3: Generate Germany CO2 emissions trend
func plotEmissions(data []Observation) error {

	p := plot.New()

	p.Title.Text = "Germany CO2 Emissions (1970-2022)"
	p.X.Label.Text = "Year"
	p.Y.Label.Text = "CO2 Emissions (MtCO2e)"

	points := make(plotter.XYs, len(data))

	for i, obs := range data {
		points[i].X = float64(obs.Year)
		points[i].Y = obs.Emissions
	}

	line, err := plotter.NewLine(points)
	if err != nil {
		return err
	}

	p.Add(line)

	err = p.Save(
		8*vg.Inch,
		5*vg.Inch,
		"germany_co2_trend.png",
	)

	if err != nil {
		return err
	}

	fmt.Println("\nEDA graph saved: germany_co2_trend.png")

	return nil
}
