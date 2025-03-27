#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script demonstrating how to use the LSTM inference module for PV power forecasting.

This script imports the LSTMLowResInference class and runs a forecast, showing
both console output and visualization options. It demonstrates how to integrate
the forecasting functionality into other applications.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from lstm_lowres_inference import LSTMLowResInference
import argparse

def run_forecast(export_csv=False, export_json=False, plot=True, save_plot=None):
    """
    Run a 24-hour PV power forecast and showcase different output options.
    
    Args:
        export_csv: Whether to export results to CSV
        export_json: Whether to export results to JSON
        plot: Whether to show the forecast plot
        save_plot: Path to save the plot (None = don't save)
    """
    print("===== PV Power Forecasting Example =====\n")
    
    # Create inference object
    forecaster = LSTMLowResInference()
    
    # Generate forecast
    print("Generating 24-hour PV power forecast...\n")
    forecast_df = forecaster.predict_next_24h()
    
    # Display the forecast in the console
    forecaster.display_forecast(forecast_df)
    
    # Export to CSV if requested
    if export_csv:
        csv_path = 'data/pv_forecast_results.csv'
        forecast_df.to_csv(csv_path)
        print(f"\nForecast exported to CSV: {csv_path}")
    
    # Export to JSON if requested
    if export_json:
        json_path = 'data/pv_forecast_results.json'
        
        # Convert to JSON-friendly format with ISO-formatted datetime index
        forecast_json = forecast_df.reset_index().to_json(
            orient='records', 
            date_format='iso'
        )
        
        # Save to file
        with open(json_path, 'w') as f:
            f.write(forecast_json)
            
        print(f"Forecast exported to JSON: {json_path}")
    
    # Plot the forecast if requested
    if plot:
        # Either save to file, or display interactively, or both
        forecaster.plot_forecast(forecast_df, save_plot)
    
    return forecast_df

def main():
    """Parse command line arguments and run the forecast."""
    parser = argparse.ArgumentParser(description='Run LSTM PV Power Forecast')
    parser.add_argument('--csv', action='store_true', help='Export results to CSV')
    parser.add_argument('--json', action='store_true', help='Export results to JSON')
    parser.add_argument('--no-plot', action='store_true', help='Don\'t show plot')
    parser.add_argument('--save-plot', type=str, help='Save plot to the specified path')
    
    args = parser.parse_args()
    
    # Create output directories if needed
    os.makedirs('data', exist_ok=True)
    
    # Run the forecast with specified options
    forecast_df = run_forecast(
        export_csv=args.csv,
        export_json=args.json,
        plot=not args.no_plot,
        save_plot=args.save_plot
    )
    
    print("\nForecast complete!")
    return forecast_df

if __name__ == "__main__":
    forecast_df = main()