import pandas as pd
import numpy as np

# Load the processed data
df = pd.read_parquet('data/processed_training_data.parquet')

# Check if isNight feature exists
print("Columns in the dataset:")
print(df.columns.tolist())
print("\nIs 'isNight' in the dataset?", 'isNight' in df.columns)

if 'isNight' in df.columns:
    # Count night vs day samples
    night_count = df['isNight'].sum()
    day_count = len(df) - night_count
    print(f"\nNight samples: {night_count} ({night_count/len(df)*100:.2f}%)")
    print(f"Day samples: {day_count} ({day_count/len(df)*100:.2f}%)")
    
    # Check correlation between isNight and power
    if 'power_w' in df.columns:
        night_power_avg = df[df['isNight'] == 1]['power_w'].mean()
        day_power_avg = df[df['isNight'] == 0]['power_w'].mean()
        print(f"\nAverage power during night: {night_power_avg}")
        print(f"Average power during day: {day_power_avg}")

# Check hour_sin and hour_cos for different times within the same hour
print("\nChecking hour_sin and hour_cos values for different times:")
# Group by hour and minute to get samples at different times
sample_times = []
for hour in range(24):
    for minute in [0, 15, 30, 45]:
        # Try to find a sample at this hour and minute
        samples = df[
            (df.index.hour == hour) & 
            (df.index.minute == minute)
        ]
        if not samples.empty:
            sample_times.append(samples.index[0])
            if len(sample_times) >= 8:  # Limit to 8 samples for clarity
                break
    if len(sample_times) >= 8:
        break

# Print the hour_sin and hour_cos values for these sample times
if sample_times:
    print("\nTime\t\thour_sin\thour_cos")
    print("-" * 40)
    for time in sample_times:
        row = df.loc[time]
        print(f"{time}\t{row['hour_sin']:.6f}\t{row['hour_cos']:.6f}")
        
    # Calculate what the values should be theoretically
    print("\nTheoretical values:")
    print("Time\t\thour_sin\thour_cos")
    print("-" * 40)
    for time in sample_times:
        hour = time.hour
        minute = time.minute
        time_of_day = hour + minute / 60
        hour_sin = np.sin(2 * np.pi * time_of_day / 24)
        hour_cos = np.cos(2 * np.pi * time_of_day / 24)
        print(f"{time}\t{hour_sin:.6f}\t{hour_cos:.6f}")
else:
    print("No samples found to check hour encoding")