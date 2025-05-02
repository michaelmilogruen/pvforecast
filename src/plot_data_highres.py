import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Helper for date formatting on plots
import numpy as np # Needed for rolling correlation

# Define the path to your parquet file
parquet_file_path = 'data/station_data_10min.parquet' # Replace if needed

# --- Configuration for Rolling Correlation ---
# Choose a window size for the rolling correlation.
# A larger window smooths more but is less reactive to short-term changes.
# 144 corresponds to 24 hours of 10-minute data.
rolling_window_size = 144

try:
    # Read the parquet file, loading both 'power_w' and 'GlobalRadiation [W m-2]' columns and the index
    print(f"Loading 'power_w' and 'GlobalRadiation [W m-2]' from {parquet_file_path} for plotting...")
    df = pd.read_parquet(parquet_file_path, columns=['power_w', 'GlobalRadiation [W m-2]'])
    print("Data loaded successfully.")

    total_size = len(df)
    if total_size == 0:
        print("Error: The DataFrame is empty, cannot plot.")
    elif 'power_w' not in df.columns or 'GlobalRadiation [W m-2]' not in df.columns:
         print("Error: Required columns ('power_w' or 'GlobalRadiation [W m-2]') not found in the file.")
    elif not pd.api.types.is_numeric_dtype(df['power_w']) or not pd.api.types.is_numeric_dtype(df['GlobalRadiation [W m-2]']):
         print("Error: 'power_w' or 'GlobalRadiation [W m-2]' are not numeric columns.")
    else:
        # Calculate the index positions for the splits
        # 70% for training, 15% for validation, 15% for testing
        train_end_idx = int(total_size * 0.7)
        val_end_idx = int(total_size * 0.85)

        # Get the actual timestamps corresponding to these index positions
        # Use min(idx, total_size - 1) to avoid index out of bounds if size is small
        train_split_timestamp = df.index[min(train_end_idx, total_size - 1)]
        val_split_timestamp = df.index[min(val_end_idx, total_size - 1)]

        print(f"Total data points: {total_size}")
        print(f"Train ends at index: {train_end_idx} (approx 70%), Timestamp: {train_split_timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"Validation ends at index: {val_end_idx} (approx 85%), Timestamp: {val_split_timestamp.strftime('%Y-%m-%d %H:%M')}")


        # --- Calculate Rolling Correlation ---
        print(f"\nCalculating rolling correlation with window size {rolling_window_size} steps...")
        # .corr() between two series inside rolling gives the rolling correlation
        rolling_corr = df['power_w'].rolling(window=rolling_window_size).corr(df['GlobalRadiation [W m-2]'])

        # Calculate overall mean correlation
        mean_corr = df['power_w'].corr(df['GlobalRadiation [W m-2]'])
        print(f"Overall mean correlation between power and radiation: {mean_corr:.4f}")


        # --- Generate Three Vertically Split Subplots ---
        print("\nGenerating Three Vertically Split Subplots...")

        # Create a figure and a set of subplots (3 rows, 1 column, sharing the x-axis)
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(15, 12))
        # axes is an array of axes objects: axes[0] (top), axes[1] (middle), axes[2] (bottom)

        # --- Top Subplot: PV Power ---
        ax1 = axes[0]
        ax1.plot(df.index, df['power_w'], color='blue', label='PV Power', alpha=0.8, linewidth=0.8)
        ax1.set_ylabel('PV Power [W]', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title('PV Power Over Time')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Add vertical lines to the top subplot
        ax1.axvline(train_split_timestamp, color='r', linestyle='--', lw=2, label='Train/Val Split')
        ax1.axvline(val_split_timestamp, color='g', linestyle='--', lw=2, label='Val/Test Split')
        # Combine and place legends
        lines1, labels1 = ax1.get_legend_handles_labels()


        # --- Middle Subplot: Global Radiation ---
        ax2 = axes[1]
        ax2.plot(df.index, df['GlobalRadiation [W m-2]'], color='orange', label='Global Radiation', alpha=0.8, linewidth=0.8)
        ax2.set_ylabel('Global Radiation [W m-2]', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.set_title('Global Radiation Over Time')
        ax2.grid(True, linestyle='--', alpha=0.6)

        # Add vertical lines to the middle subplot
        ax2.axvline(train_split_timestamp, color='r', linestyle='--', lw=2) # No label needed, legend is shared
        ax2.axvline(val_split_timestamp, color='g', linestyle='--', lw=2) # No label needed
        lines2, labels2 = ax2.get_legend_handles_labels()


        # --- Bottom Subplot: Rolling Correlation ---
        ax3 = axes[2]
        ax3.plot(rolling_corr.index, rolling_corr, color='purple', label=f'Rolling Corr (window={rolling_window_size})', alpha=0.8, linewidth=0.8)
        ax3.set_ylabel('Correlation Coefficient', color='purple')
        ax3.tick_params(axis='y', labelcolor='purple')
        ax3.set_title('Rolling Correlation (Power vs. Global Radiation)')
        ax3.set_xlabel('Timestamp') # X-axis label only on the bottom plot

        # Add horizontal line for overall mean correlation
        ax3.axhline(mean_corr, color='gray', linestyle='-', lw=2, label=f'Overall Mean Corr ({mean_corr:.2f})')

        # Add vertical lines to the bottom subplot
        ax3.axvline(train_split_timestamp, color='r', linestyle='--', lw=2) # No label needed
        ax3.axvline(val_split_timestamp, color='g', linestyle='--', lw=2) # No label needed
        ax3.grid(True, linestyle='--', alpha=0.6)
        lines3, labels3 = ax3.get_legend_handles_labels() # Get legend handles for the correlation plot


        # Add a single legend for all plots (optional, can be placed on one of the axes instead)
        # fig.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right', bbox_to_anchor=(1.05, 1))

        # Alternative: place legend on the bottom plot (common with sharex)
        ax3.legend(loc='upper left')


        # Improve date formatting on the shared x-axis
        fig.autofmt_xdate() # Auto-format dates

        # Add a main title for the entire figure if desired
        fig.suptitle('Data Time Series and Rolling Correlation with Data Splits', y=1.02, fontsize=16) # Adjust y for position


        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make room for suptitle
        plt.show() # Display the plot

        print("Three vertically split subplots displayed: PV Power, Global Radiation, and Rolling Correlation with data splits.")


except FileNotFoundError:
    print(f"Error: The file was not found at {parquet_file_path}")
except ImportError:
    print("Error: pandas or matplotlib not installed.")
    print("Please install them using: pip install pandas matplotlib pyarrow")
except Exception as e:
    print(f"An unexpected error occurred: {e}")