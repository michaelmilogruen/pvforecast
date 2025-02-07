import pandas as pd
import os
from datetime import datetime

def calculate_power(energy_wh):
    """Convert energy in Wh per 5 minutes to average power in watts"""
    return energy_wh * 12  # multiply by 12 to convert from Wh/5min to watts

def merge_yearly_data():
    """Merge data files from 2022-2025 chronologically and add power column"""
    data_dir = "data"
    years = range(2022, 2026)
    all_data = []
    
    print("Reading data files...")
    for year in years:
        file_path = os.path.join(data_dir, f"data_{year}.xlsx")
        print(f"Processing data_{year}.xlsx...")
        
        # Skip the first row which contains format string
        df = pd.read_excel(file_path, skiprows=1)
        
        # Rename columns to match expected names
        df.columns = ['Datum und Uhrzeit', 'Energie | Symo 12.5-3-M (2)', 'PV Produktion']
        
        # Add year column for reference
        df['Year'] = year
        
        # Add power column (converting from Wh/5min to watts)
        energy_column = 'Energie | Symo 12.5-3-M (2)'
        df['Power (W)'] = df[energy_column].apply(calculate_power)
        
        # Convert date column to datetime with correct format
        df['Datum und Uhrzeit'] = pd.to_datetime(df['Datum und Uhrzeit'], 
                                               format='%d.%m.%Y %H:%M')
        
        all_data.append(df)
    
    # Combine all data
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by date
    merged_df = merged_df.sort_values('Datum und Uhrzeit')
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(data_dir, f'merged_yearly_data_{timestamp}.xlsx')
    print(f"\nSaving merged data to {output_file}...")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        merged_df.to_excel(writer, sheet_name='All_Years_Combined', index=False)
        
        # Also save individual years as separate sheets
        for year in years:
            year_data = merged_df[merged_df['Year'] == year]
            year_data.to_excel(writer, sheet_name=f'Data_{year}', index=False)
    
    print("Merge completed successfully!")
    print(f"Total rows in merged data: {len(merged_df)}")
    
    # Print sample of data to verify power calculation
    print("\nSample of merged data with power calculation:")
    print(merged_df[['Datum und Uhrzeit', energy_column, 'Power (W)', 'Year']].head())

if __name__ == "__main__":
    merge_yearly_data()