import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
from datetime import datetime

def plot_pv_power(csv_file, output_dir='figures/pv_power'):
    """
    Plot PV power data from CSV file
    
    Args:
        csv_file (str): Path to CSV file
        output_dir (str): Directory to save the output plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV file with semicolon separator
    # Skip the first two rows (header and units)
    df = pd.read_csv(csv_file, sep=';', decimal=',', encoding='utf-8', skiprows=2)
    
    # Rename columns for easier handling
    df.columns = ['timestamp', 'energy_symo', 'pv_production', 'pv_power']
    
    # Convert timestamp column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M')
    
    # Convert values in pv_power column to float
    # Replace invalid values with NaN
    df['pv_power'] = pd.to_numeric(df['pv_power'], errors='coerce')
    
    # Filter out rows where timestamp or pv_power is NaN
    df = df.dropna(subset=['timestamp', 'pv_power'])

    # Get date range for title
    start_date = df['timestamp'].min().strftime('%Y-%m-%d')
    end_date = df['timestamp'].max().strftime('%Y-%m-%d')
    
    # Create figure and axis
    plt.figure(figsize=(12, 6))
    
    # Plot PV power
    plt.plot(df['timestamp'], df['pv_power'], 'b-', linewidth=1)
    
    # Format x-axis to show monthly ticks
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()  # Rotate date labels
    
    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('PV Power (W)')
    plt.title(f'PV Power Production ({start_date} to {end_date})')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, f'pv_daily_profile_{start_date.replace("-", "")}_to_{end_date.replace("-", "")}.png')
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    plot_pv_power('data/raw_data_evt_act/merge.CSV')