import pandas as pd
import io
import os # Import os to check for file existence

# --- Configuration ---
# Option 1: Define the path to your CSV file
csv_file_path = 'data/raw_data_evt_act/merge.CSV'
# Option 2: Or, paste your CSV data directly into this string variable
# (Make sure the file path option is commented out or the file doesn't exist if using this)


# --- Function to Load and Process Data ---
def process_pv_data(source):
    """
    Reads PV data from a file path or string buffer, cleans it,
    and filters it to include only timestamps exactly at 10 minutes past the hour.

    Args:
        source (str or io.StringIO): Either a file path to the CSV
                                     or an io.StringIO object containing the CSV data.

    Returns:
        pandas.DataFrame: The filtered DataFrame containing only HH:10 timestamps,
                          or None if reading fails.
                          Returns the original cleaned DataFrame as a second element
                          in the tuple for comparison.
    """
    try:
        # Read the first line to get header names
        if isinstance(source, str): # If it's a file path
            with open(source, 'r', encoding='utf-8') as f:
                header_line = f.readline().strip()
                # Skip the second line (units)
                f.readline()
                # Read the rest of the data
                df = pd.read_csv(f,
                                 sep=';',
                                 header=None, # Header is already read
                                 decimal=',',
                                 skipinitialspace=True,
                                 na_values=[' ']) # Treat single spaces as NaN
        elif isinstance(source, io.StringIO): # If it's a string buffer
            header_line = source.readline().strip()
            source.readline() # Skip units line
            df = pd.read_csv(source,
                             sep=';',
                             header=None,
                             decimal=',',
                             skipinitialspace=True,
                             na_values=[' '])
        else:
            print("Error: Invalid source type. Must be file path or io.StringIO.")
            return None, None

        # Assign column names extracted from the first line
        column_names = [name.strip() for name in header_line.split(';')]
        df.columns = column_names

        # --- Data Cleaning ---
        # Strip whitespace from all string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()

        # Rename columns for easier access
        df.rename(columns={
            'Datum und Uhrzeit': 'Timestamp',
            'Energie | Symo 12.5-3-M (2)': 'Energy_Symo',
            'PV Produktion': 'PV_Production_Wh',
            'PV Leistung': 'PV_Power_W'
        }, inplace=True)

        # Convert 'Timestamp' column to datetime objects
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M', errors='coerce')

        # Drop rows where timestamp conversion failed
        df.dropna(subset=['Timestamp'], inplace=True)

        # Set 'Timestamp' as the index
        df.set_index('Timestamp', inplace=True)

        # Convert numeric columns to numeric types
        numeric_cols = ['Energy_Symo', 'PV_Production_Wh', 'PV_Power_W']
        for col in numeric_cols:
            if df[col].dtype == 'object':
                 df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Optional: Drop rows with NaN in numeric columns after conversion
        # df.dropna(subset=numeric_cols, inplace=True)

        print("--- Original Data (Cleaned) Head ---")
        print(df.head())
        print("\n--- Original Data Info ---")
        df.info()
        print("-" * 30)


        # --- Filtering ---
        # Select only the rows where the minute of the timestamp is exactly 10
        df_filtered = df[df.index.minute == 10].copy() # Use .copy() to avoid SettingWithCopyWarning

        return df_filtered, df # Return filtered and original cleaned df

    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return None, None

# --- Main Execution ---
if __name__ == "__main__":
    data_source = None
    # Check if the file exists and use it, otherwise use the string data
    if os.path.exists(csv_file_path):
        print(f"Reading data from file: {csv_file_path}")
        data_source = csv_file_path
    elif 'csv_data' in globals() and csv_data:
        print("Reading data from the 'csv_data' variable.")
        data_source = io.StringIO(csv_data)
    else:
        print("Error: No data source specified. Set 'csv_file_path' or 'csv_data'.")

    if data_source:
        filtered_df, original_df = process_pv_data(data_source)

        if filtered_df is not None:
            print("\n--- Filtered DataFrame (Timestamps at HH:10) ---")
            print(filtered_df)

            # Save the filtered data to a new CSV file
            output_dir = 'data/raw_data_evt_act'
            output_filename = os.path.join(output_dir, 'merge_filtered.csv')
            
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            filtered_df.to_csv(output_filename, sep=';', decimal=',')
            print(f"\nFiltered data saved to {output_filename}")

