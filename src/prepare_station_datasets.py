import pandas as pd
import numpy as np
from datetime import datetime
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pvlib
from pvlib.location import Location
from pvlib.clearsky import simplified_solis
from sklearn.preprocessing import MinMaxScaler, StandardScaler # Still included, but not used in this version - good for future scaling step
import joblib # Still included, but not used in this version - good for saving scalers later
import matplotlib.pyplot as plt
import seaborn as sns
import logging # Using logging instead of print for better output control

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StationDataProcessor:
    def __init__(self, chunk_size: int = 100000):
        """
        Initialize the data processor with configuration and logging.
        Includes location and PV panel parameters.
        """
        self.chunk_size = chunk_size
        self.high_res_frequency = '10min' # Original station data frequency
        self.low_res_frequency = '1h'    # Target low resolution frequency

        # Location parameters for Leoben (updated to provided simpler values)
        self.latitude = 47.3877
        self.longitude = 15.0941
        self.altitude = 541  # approximate altitude for Leoben
        self.tz_local = 'Europe/Vienna' # Use a timezone name that handles DST
        self.tz_utc = 'UTC'

        # PV Panel Parameters (provided by user)
        self.surface_tilt = 30 # degrees from horizontal
        self.surface_azimuth = 149.716 # degrees from North (0=North, 90=East, 180=South, 270=West)

        # Data Quality Thresholds (These are now primarily for reference or potential future use)
        self.IRRADIATION_THRESHOLD_DAYTIME = 20  # W/m²
        self.POWER_ZERO_THRESHOLD_DAYTIME = 1    # W
        self.CSI_LOW_THRESHOLD_DAYTIME = 0.05    # Min acceptable Clear Sky Index
        self.CSI_HIGH_THRESHOLD_DAYTIME = 1.5    # Max acceptable Clear Sky Index

        # Pvlib parameters defaults (can be refined)
        self.aod700_default = 0.1
        self.precipitable_water_default = 1.0
        self.temperature_default = 12.0 # Default temp if station data is missing

        # Scalers (not used in this version, but kept for potential future scaling step)
        self.minmax_scaler = None
        self.standard_scaler = None

    def import_pv_data(self, file_path: str) -> pd.DataFrame:
        """
        Import PV data from a semicolon-separated CSV file with European decimal format.

        Args:
            file_path: Path to the CSV file

        Returns:
            pd.DataFrame: Processed DataFrame with proper types and datetime index
        """
        logging.info(f"Importing PV data from {file_path}")

        # Read the file with proper delimiter and decimal separator
        # Skip the second row (units)
        df = pd.read_csv(file_path,
                         sep=';',
                         decimal=',',
                         skiprows=[1],  # Skip the row with units
                         na_values=['n/a', '#WERT!'])  # Handle n/a and #WERT! as NaN

        # Debug: Print some statistics about raw data before processing
        logging.info(f"Raw PV data shape: {df.shape}")
        logging.info(f"Raw PV data columns: {df.columns.tolist()}")
        logging.info(f"Sample of raw PV data 'PV Leistung' column: {df['PV Leistung'].head(10).tolist()}")
        logging.info(f"Number of non-zero 'PV Leistung' values: {(pd.to_numeric(df['PV Leistung'], errors='coerce') > 0).sum()}")

        # Convert timestamp to datetime and set as index
        df['Datum und Uhrzeit'] = pd.to_datetime(df['Datum und Uhrzeit'],
                                                 format='%d.%m.%Y %H:%M')
        df = df.set_index('Datum und Uhrzeit')

        # Rename columns to more standardized names
        df = df.rename(columns={
            'Energie | Symo 12.5-3-M (2)': 'energy_wh',
            'PV Produktion': 'energy_production_wh',
            'PV Leistung': 'power_w'
        })

        # Convert all columns to numeric (just to be sure)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Debug: Check power_w values after conversion
        logging.info(f"Number of non-zero power_w values after conversion: {(df['power_w'] > 0).sum()}")

        # Make index timezone-aware to match station data
        # Assume data is in local time (Europe/Vienna) and convert to UTC
        local_tz = 'Europe/Vienna'
        df = df.tz_localize(local_tz, ambiguous='NaT', nonexistent='NaT')

        # Drop rows where localization failed (ambiguous times during DST transitions)
        if df.index.isna().any():
            na_count = df.index.isna().sum()
            logging.warning(f"Dropping {na_count} rows with ambiguous timestamps during localization")
            df = df.loc[~df.index.isna()]

        # Convert to UTC to match station data
        df = df.tz_convert('UTC')

        # Debug: Check power_w values after timezone conversion
        logging.info(f"Number of non-zero power_w values after timezone conversion: {(df['power_w'] > 0).sum()}")

        logging.info(f"Imported PV data: {len(df)} rows, columns: {df.columns.tolist()}")
        logging.info(f"Time range: {df.index.min()} to {df.index.max()}")
        logging.info(f"PV data index timezone: {df.index.tz}")

        return df

    def read_csv_in_chunks(self, filepath: str, date_column: str,
                           date_format: str,
                           skiprows: int = 0, sep: str = ',',
                           decimal: str = '.', encoding: str = 'utf-8',
                           tz_localize: str = None) -> pd.DataFrame:
        """
        Read CSV file in chunks, process datetime, and set index.
        Handles potential issues with combined date/time columns if sep is incorrect.
        Includes optional timezone localization.
        """
        logging.info(f"Reading {filepath} in chunks...")
        chunks = []

        # Use the correct separator from the start
        try:
            for chunk in pd.read_csv(filepath, sep=sep, chunksize=self.chunk_size,
                                     skiprows=skiprows, decimal=decimal,
                                     encoding=encoding, low_memory=False): # low_memory=False can help with mixed types

                # Convert datetime
                # Use dayfirst=True for formats like dd.mm.yyyy if needed, but format string is more robust
                chunk[date_column] = pd.to_datetime(chunk[date_column], format=date_format, errors='coerce')

                # Drop rows where datetime conversion failed
                initial_rows = len(chunk)
                chunk.dropna(subset=[date_column], inplace=True)
                if len(chunk) < initial_rows:
                    logging.warning(f"Dropped {initial_rows - len(chunk)} rows due to invalid date format in chunk.")

                # Set datetime as index
                chunk.set_index(date_column, inplace=True)

                # Localize index if requested
                if tz_localize:
                    # Use ambiguous='NaT', nonexistent='NaT' to handle DST transitions gracefully
                    chunk.index = chunk.index.tz_localize(tz_localize, ambiguous='NaT', nonexistent='NaT')
                    if chunk.index.isna().any():
                        num_na_tz = chunk.index.isna().sum()
                        logging.warning(f"Dropped {num_na_tz} rows in chunk due to ambiguous/nonexistent timestamps during localization to {tz_localize}.")
                        chunk.dropna(inplace=True) # Drop rows that couldn't be localized


                chunks.append(chunk)

            df = pd.concat(chunks)
            logging.info(f"Finished reading {filepath}. Total shape: {df.shape}")
            return df

        except Exception as e:
            logging.error(f"Error reading CSV file {filepath}: {e}")
            # Attempt to read a small portion for debugging column issues
            try:
                debug_df = pd.read_csv(filepath, sep=sep, nrows=5, skiprows=skiprows, encoding=encoding)
                logging.info(f"First few rows read with sep='{sep}':\n{debug_df}")
                # Try reading with different separator if applicable (e.g., comma vs semicolon)
                if sep == ',':
                    debug_df_alt = pd.read_csv(filepath, sep=';', nrows=5, skiprows=skiprows, encoding=encoding)
                    logging.info(f"First few rows read with sep=';':\n{debug_df_alt}")
                elif sep == ';':
                    debug_df_alt = pd.read_csv(filepath, sep=',', nrows=5, skiprows=skiprows, encoding=encoding)
                    logging.info(f"First few rows read with sep=',' :\n{debug_df_alt}")

            except Exception as debug_e:
                logging.error(f"Could not perform debug read: {debug_e}")

            raise # Re-raise the original error

    def process_station_data(self, filepath: str) -> pd.DataFrame:
        """Process local weather station data (10min). Quality flag filtering is removed."""
        logging.info(f"Processing station data from {filepath}...")

        # Station data format has +00:00, implying UTC. Localize to UTC.
        # Quality flag filtering is removed as requested.
        df = self.read_csv_in_chunks(
            filepath=filepath,
            date_column='time',
            date_format='%Y-%m-%dT%H:%M+00:00',
            sep=',',
            tz_localize=self.tz_utc # Localize index to UTC
        )

        logging.info(f"Raw station data shape: {df.shape}")

        # Define critical measurements and their quality flags with descriptive names
        # Quality flag filtering is removed here.
        critical_cols_flags = {
            'cglo': 'cglo_flag', # Global radiation
            'tl': 'tl_flag',     # Air temperature
            'ff': 'ff_flag'      # Wind speed
        }

        # Select relevant columns (including original flag columns if present, though not used for filtering)
        relevant_cols = ['cglo', 'tl', 'ff', 'rr', 'p'] + list(critical_cols_flags.values())
        available_cols = [col for col in relevant_cols if col in df.columns]
        df = df[available_cols].copy() # Use .copy()

        # Rename columns to be more descriptive
        column_mapping = {
            'cglo': 'GlobalRadiation [W m-2]', # Global radiation measurement
            'tl': 'Temperature [degree_Celsius]', # Air temperature
            'ff': 'WindSpeed [m s-1]', # Wind speed
            'rr': 'Precipitation [mm]', # Rainfall/precipitation
            'p': 'Pressure [hPa]' # Air pressure
        }
        # Only rename columns that exist in the dataframe
        column_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=column_mapping).copy() # Use .copy()

        logging.info(f"Processed station data shape after initial renaming: {df.shape}")
        logging.info("Sample of processed station data:")
        logging.info(df.head())
        logging.info(f"Station data index timezone: {df.index.tz}")


        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features for machine learning."""
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
             logging.warning("Index is not DatetimeIndex. Cannot add time features.")
             return df.copy()

        # Perform calculations in UTC if index is timezone-aware UTC, otherwise use naive
        # Convert to UTC if timezone-aware, otherwise use as is
        temp_index = df.index
        if temp_index.tz is not None:
            try:
                temp_index = temp_index.tz_convert(self.tz_utc)
                # logging.info("Calculating time features using UTC index.") # Suppress this frequent log
            except Exception as e:
                 logging.warning(f"Could not convert index to UTC for time feature calculation: {e}. Using original index.")
                 # Fallback to original index if conversion fails


        df_out = df.copy() # Create a copy to add features to

        # Use the potentially converted index for feature calculation
        df_out['hour'] = temp_index.hour + temp_index.minute / 60
        df_out['day_of_year'] = temp_index.dayofyear

        # Add circular encoding for hour (sin/cos)
        angle_hour = 2 * np.pi * df_out['hour'] / 24
        df_out['hour_sin'] = np.sin(angle_hour)
        df_out['hour_cos'] = np.cos(angle_hour)

        # Add circular encoding for day of year (sin/cos)
        angle_day = 2 * np.pi * df_out['day_of_year'] / 365.25 # Account for leap years
        df_out['day_sin'] = np.sin(angle_day)
        df_out['day_cos'] = np.cos(angle_day)

        # logging.info("Added time-based features.") # Suppress this frequent log
        return df_out.copy() # Return a copy

    def add_irradiation_night_mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add isNight mask based on irradiation values.
        We consider it night when global radiation is below a threshold.
        """
        # Define threshold for night (very low irradiation)
        NIGHT_THRESHOLD = 10 # W/m²

        df_out = df.copy() # Create a copy to add feature to

        # Create isNight mask (1 for night, 0 for day) based on radiation
        if 'GlobalRadiation [W m-2]' in df_out.columns:
            # Ensure the column is numeric before comparison
            df_out['GlobalRadiation [W m-2]'] = pd.to_numeric(df_out['GlobalRadiation [W m-2]'], errors='coerce').fillna(0)
            df_out['isNight_irradiation'] = (df_out['GlobalRadiation [W m-2]'] < NIGHT_THRESHOLD).astype(int)
            logging.info(f"Created isNight mask based on GlobalRadiation < {NIGHT_THRESHOLD} W/m².")
        else:
            logging.warning("Warning: GlobalRadiation column not found. Cannot create irradiation-based isNight mask.")
            # Don't create the column if source is missing

        return df_out.copy() # Return a copy


    def calculate_solar_geometry_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates solar position (zenith, azimuth), clear sky irradiance,
        Clear Sky Index, Angle of Incidence (AOI), and solar night mask using pvlib.
        Input df is expected to have a timezone-aware index (UTC).
        Uses measured pressure if available, otherwise calculates from altitude.
        """
        logging.info("Calculating solar geometry features (solar position, clear sky, AOI)...")

        # Ensure input index is timezone-aware
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
             logging.error("Input DataFrame index is not timezone-aware. Cannot calculate solar geometry.")
             # Add placeholder columns filled with NaN if they don't exist
             df_out = df.copy()
             for col in ['SolarZenith [degrees]', 'SolarAzimuth [degrees]', 'AOI [degrees]',
                          'ClearSkyGHI', 'ClearSkyDNI', 'ClearSkyDHI', 'isNight_solar', 'ClearSkyIndex']:
                 if col not in df_out.columns:
                     df_out[col] = np.nan
             return df_out # Return original df with added NaN columns


        # Create a temporary timezone-aware index in the local timezone for pvlib calculations
        # Handle ambiguous/nonexistent times during DST transitions by marking them NaT
        temp_index_local = df.index.tz_convert(self.tz_local)

        if temp_index_local.isna().any():
             num_na_tz = temp_index_local.isna().sum()
             logging.warning(f"Found {num_na_tz} timestamps that failed timezone conversion to {self.tz_local} (e.g., during DST). These rows will be skipped for pvlib calculations.")
             # Select rows where localization was successful - Use the original UTC index for slicing
             valid_utc_index = temp_index_local.dropna().index # This index is still UTC because it came from df.index
             df_pvlib_subset = df.loc[valid_utc_index].copy() # Subset the original df by the valid UTC index
             temp_index_local_clean = temp_index_local.dropna() # Get the cleaned LOCALIZED index for pvlib calls
             logging.info(f"Pvlib calculations will run on a subset of {len(df_pvlib_subset)} rows.")
        else:
            df_pvlib_subset = df.copy() # Use the whole df if no localization issues
            temp_index_local_clean = temp_index_local # Use the whole LOCALIZED index
            logging.info(f"Pvlib calculations will run on all {len(df_pvlib_subset)} rows.")


        if df_pvlib_subset.empty:
             logging.warning("No data available for pvlib calculations after timezone conversion and subsetting.")
             # Add placeholder columns filled with NaN if they don't exist, indexed like the original df
             df_out = df.copy()
             for col in ['SolarZenith [degrees]', 'SolarAzimuth [degrees]', 'AOI [degrees]',
                          'ClearSkyGHI', 'ClearSkyDNI', 'ClearSkyDHI', 'isNight_solar', 'ClearSkyIndex']:
                 if col not in df_out.columns:
                     df_out[col] = np.nan
             return df_out # Return original df with added NaN columns


        # Get pressure and temperature for solar position calculation and clear sky model
        # Use measured pressure if available, otherwise calculate from altitude
        # Use measured temperature if available, otherwise use default
        if 'Pressure [hPa]' in df_pvlib_subset.columns:
            # Use actual pressure from the subset, convert hPa to Pa, fill NaNs with altitude-derived pressure
            pressure_pa = pd.to_numeric(df_pvlib_subset['Pressure [hPa]'], errors='coerce') * 100 # Convert hPa to Pa
            # Fill NaNs in measured pressure with altitude-derived pressure
            pressure_pa = pressure_pa.fillna(pvlib.atmosphere.alt2pres(self.altitude) * 100)
            logging.info("Using measured pressure for solar position and clear sky calculations.")
        else:
            # Calculate pressure from altitude if pressure column is missing
            pressure_pa = pvlib.atmosphere.alt2pres(self.altitude) * 100 # Calculate pressure from altitude (Pa)
            # If pressure column is missing, create a Series with the altitude-derived pressure for all timestamps
            pressure_pa = pd.Series(pressure_pa, index=temp_index_local_clean)
            logging.warning(f"Pressure column not found in subset. Using altitude-derived pressure ({pressure_pa.mean()} Pa) for solar position and clear sky calculations.")


        if 'Temperature [degree_Celsius]' in df_pvlib_subset.columns:
             # Use actual temperature from the subset, fill NaNs with default
             temperature_c = pd.to_numeric(df_pvlib_subset['Temperature [degree_Celsius]'], errors='coerce').fillna(self.temperature_default)
             logging.info("Using measured temperature for solar position calculation.")
        else:
             temperature_c = self.temperature_default
             # If temperature column is missing, create a Series with the default temperature for all timestamps
             temperature_c = pd.Series(temperature_c, index=temp_index_local_clean)
             logging.warning(f"Temperature column not found in subset. Using default temperature ({self.temperature_default}°C) for solar position calculation.")


        # Get solar position (zenith, azimuth) - Pass the CLEAN LOCALIZED index subset
        # Use the pressure and temperature series (which have the same index as temp_index_local_clean)
        solar_position = pvlib.solarposition.get_solarposition(
            time=temp_index_local_clean, # Pass the cleaned, localized index
            latitude=self.latitude,
            longitude=self.longitude,
            altitude=self.altitude, # Altitude is still needed for some internal pvlib calculations
            pressure=pressure_pa, # Pass pressure in Pa (Series)
            temperature=temperature_c # Pass temperature in °C (Series)
            # method='nrel_numpy' is default
        )
        logging.info(f"Calculated solar position for {len(solar_position)} timestamps.")


        # --- Calculate Angle of Incidence (AOI) ---
        aoi_series = pd.Series(np.nan, index=temp_index_local_clean) # Initialize with NaNs, same index as subset
        if not solar_position.empty:
             # Ensure solar position angles are numeric (they should be, but as a safeguard)
             solar_position['zenith'] = pd.to_numeric(solar_position['zenith'], errors='coerce')
             solar_position['azimuth'] = pd.to_numeric(solar_position['azimuth'], errors='coerce')

             # Calculate AOI directly into a series - relies on solar_position index
             # Only calculate where zenith and azimuth are not NaN
             valid_solar_pos_mask = solar_position[['zenith', 'azimuth']].notna().all(axis=1)
             if valid_solar_pos_mask.any():
                 aoi_series_valid = pvlib.irradiance.aoi(
                     surface_tilt=self.surface_tilt,
                     surface_azimuth=self.surface_azimuth,
                     solar_zenith=solar_position.loc[valid_solar_pos_mask, 'zenith'],
                     solar_azimuth=solar_position.loc[valid_solar_pos_mask, 'azimuth']
                 )
                 aoi_series.loc[aoi_series_valid.index] = aoi_series_valid # Assign results back to the full series
                 logging.info(f"Calculated Angle of Incidence (AOI) series for {len(aoi_series_valid)} timestamps.")
             else:
                 logging.warning("No valid solar zenith or azimuth found to calculate AOI.")

        else:
            logging.warning("Solar position calculation failed or returned empty. AOI series will be all NaN.")


        # --- Calculate Clear Sky Irradiance ---
        clear_sky = pd.DataFrame(index=temp_index_local_clean) # Initialize with NaNs, same index as subset
        night_mask_solar_series = pd.Series(np.nan, index=temp_index_local_clean) # Initialize
        if not solar_position.empty:
             apparent_elevation = 90 - solar_position['apparent_zenith']
             apparent_elevation = apparent_elevation.clip(lower=0)

             # Pass the pressure series (pressure_pa) to simplified_solis
             clear_sky = simplified_solis(
                 apparent_elevation=apparent_elevation, # uses solar_position index
                 aod700=self.aod700_default,
                 precipitable_water=self.precipitable_water_default,
                 pressure=pressure_pa # Pass the pressure Series in Pa
             )

             # Set irradiance components to 0 for nighttime (apparent elevation <= 0)
             night_mask_solar_series = (apparent_elevation <= 0) # Capture the mask series
             for col in ['ghi', 'dni', 'dhi']:
                 if col in clear_sky.columns:
                     # Ensure clear sky columns are numeric before masking and fill NaNs
                     clear_sky[col] = pd.to_numeric(clear_sky[col], errors='coerce').fillna(0)
                     clear_sky.loc[night_mask_solar_series, col] = 0.0
                 else:
                     logging.warning(f"Clear sky column '{col}' not found in pvlib output.")
                     clear_sky[col] = 0.0 # Add the column with zeros if missing from pvlib output
             logging.info("Calculated clear sky irradiance and solar night mask series.")

        else:
             logging.warning("Solar position is empty. Cannot calculate clear sky irradiance or solar night mask.")
             # Create empty clear_sky dataframe and night mask series with the same index as the subset used for pvlib calls
             for col in ['ghi', 'dni', 'dhi']:
                 clear_sky[col] = np.nan
             # night_mask_solar_series already initialized to NaNs


        # --- Combine calculated pvlib features into a single DataFrame ---
        # These dataframes/series (solar_position, clear_sky, aoi_series, night_mask_solar_series)
        # have the *temporary localized* index (temp_index_local_clean).
        # Create a dataframe with all the desired pvlib outputs, indexed by temp_index_local_clean.

        # Start with a dataframe using the common index subset
        calculated_pvlib_features = pd.DataFrame(index=temp_index_local_clean)

        # Add solar position columns
        if not solar_position.empty:
             calculated_pvlib_features['SolarZenith [degrees]'] = solar_position['zenith']
             calculated_pvlib_features['SolarAzimuth [degrees]'] = solar_position['azimuth']
        else:
             for col in ['SolarZenith [degrees]', 'SolarAzimuth [degrees]']:
                 calculated_pvlib_features[col] = np.nan

        # Add clear sky columns
        if clear_sky is not None and not clear_sky.empty:
             calculated_pvlib_features['ClearSkyGHI'] = clear_sky['ghi']
             calculated_pvlib_features['ClearSkyDNI'] = clear_sky['dni']
             calculated_pvlib_features['ClearSkyDHI'] = clear_sky['dhi']
        else:
             for col in ['ClearSkyGHI', 'ClearSkyDNI', 'ClearSkyDHI']:
                 calculated_pvlib_features[col] = np.nan

        # Add AOI series
        if aoi_series is not None and not aoi_series.empty:
             calculated_pvlib_features['AOI [degrees]'] = aoi_series
        else:
             calculated_pvlib_features['AOI [degrees]'] = np.nan

        # Add solar night mask series
        if night_mask_solar_series is not None and not night_mask_solar_series.empty:
             calculated_pvlib_features['isNight_solar'] = night_mask_solar_series.astype(int)
        else:
             # If mask calculation failed, fill with NaN or a default (e.g., 0 for "unknown")
             calculated_pvlib_features['isNight_solar'] = np.nan # Keep as NaN if could not be determined


        # Convert the index of calculated_pvlib_features back to timezone-aware UTC
        # This is the timezone of the original input df
        if not calculated_pvlib_features.empty:
             # Ensure index is timezone-aware before converting
             if calculated_pvlib_features.index.tz is None:
                 logging.warning("Index of calculated_pvlib_features is unexpectedly naive. Attempting localization before converting to UTC.")
                 # Try localizing to the source (local) timezone first
                 try:
                     calculated_pvlib_features.index = calculated_pvlib_features.index.tz_localize(self.tz_local, ambiguous='NaT', nonexistent='NaT')
                     if calculated_pvlib_features.index.isna().any():
                          logging.warning("NaNs introduced during re-localization of pvlib features index.")
                          # Handle these NaNs if necessary, e.g., drop rows, but might lose alignment
                          # For now, proceed, conversion to UTC might handle some.
                 except Exception as e:
                     logging.error(f"Failed to re-localize pvlib features index to {self.tz_local}: {e}")
                     logging.warning("Proceeding with potential timezone issues.")

             # Now convert to UTC
             calculated_pvlib_features.index = calculated_pvlib_features.index.tz_convert(self.tz_utc)
             calculated_pvlib_features.index.name = 'datetime' # Ensure index name matches for merge
        else:
             # If the dataframe is empty, give the index a name and timezone for consistency before merge
             calculated_pvlib_features.index.name = 'datetime'
             calculated_pvlib_features = calculated_pvlib_features.tz_localize(self.tz_utc) # Make empty index UTC-aware


        # --- Merge the calculated pvlib features back to the original dataframe index ---
        # Use how='left' on df to keep all original timestamps from the input df (which is UTC-aware)
        # The calculated features dataframe is also UTC-aware now.
        df_merged_pvlib = df.merge(calculated_pvlib_features, left_index=True, right_index=True, how='left').copy() # Use .copy()


        # --- Calculate Clear Sky Index (after merging clear sky GHI) ---
        # This needs GlobalRadiation from the original station data AND ClearSkyGHI
        if 'GlobalRadiation [W m-2]' in df_merged_pvlib.columns and 'ClearSkyGHI' in df_merged_pvlib.columns:
            # Ensure columns are numeric (they should be after initial numeric conversion, but double check)
            df_merged_pvlib['GlobalRadiation [W m-2]'] = pd.to_numeric(df_merged_pvlib['GlobalRadiation [W m-2]'], errors='coerce')
            df_merged_pvlib['ClearSkyGHI'] = pd.to_numeric(df_merged_pvlib['ClearSkyGHI'], errors='coerce')

            # Handle division by zero and very small values
            # Fill NaNs before division
            rad_measured = df_merged_pvlib['GlobalRadiation [W m-2]'].fillna(0)
            cs_ghi = df_merged_pvlib['ClearSkyGHI'].fillna(0)

            df_merged_pvlib['ClearSkyIndex'] = np.where(
                cs_ghi > 1.0, # Only calculate for significant clear sky irradiance
                rad_measured / (cs_ghi + np.finfo(float).eps), # Add epsilon to denominator
                0.0 # Set to 0 for nighttime or very low clear sky irradiance
            )

            # Re-added: Clipping unrealistic clear sky index values
            df_merged_pvlib['ClearSkyIndex'] = df_merged_pvlib['ClearSkyIndex'].clip(0, self.CSI_HIGH_THRESHOLD_DAYTIME) # Clip upper end based on defined threshold
            logging.info(f"Calculated Clear Sky Index (clipped at {self.CSI_HIGH_THRESHOLD_DAYTIME}).")
        else:
             logging.warning("Cannot calculate Clear Sky Index: Missing 'GlobalRadiation [W m-2]' or 'ClearSkyGHI'.")
             if 'ClearSkyIndex' not in df_merged_pvlib.columns:
                  df_merged_pvlib['ClearSkyIndex'] = np.nan


        logging.info("Finished calculating solar geometry features.")
        return df_merged_pvlib.copy() # Return a copy


    def process_pv_data(self, filepath: str) -> pd.DataFrame:
        """
        Process PV production data. Uses explicit column names based on user info.
        """
        logging.info(f"\nProcessing PV data from {filepath}...")

        # Explicit mapping for expected column names in the input CSV to standardized names
        # Add other relevant PV columns if known, mapping to their desired standardized names
        explicit_pv_mapping = {
            'Datum und Uhrzeit': 'datetime', # Timestamp column
            'PV Leistung': 'power_w',        # Power column in Watts
            'Energie | Symo 12.5-3-M (2)': 'energy_wh_symo', # Example for an energy column
            'PV Produktion': 'energy_production_wh' # Another example for an energy column
        }

        # Read the data, skipping header/units row (assuming the row after the header is units)
        # Use ';' separator and decimal comma
        df = pd.read_csv(filepath, sep=';', skiprows=1, decimal=',',
                         thousands=None, na_values=['n/a', '#WERT!', '', ' '], # Handle common missing values
                         encoding='utf-8', low_memory=False)

        logging.info("First few rows of raw PV data:")
        logging.info(df.head())

        # Check and remove the units row if it exists (contains [dd.MM.yyyy HH:mm], [Wh], etc.)
        # Assuming the date column is the first one (index 0)
        initial_rows = len(df)
        if initial_rows > 0 and df.iloc[0, 0] and isinstance(df.iloc[0, 0], str) and '[' in df.iloc[0, 0] and ']' in df.iloc[0, 0]:
             df = df.iloc[1:].copy() # Skip the first row after header
             logging.info("Skipped units row in PV data.")


        # Strip any leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()
        logging.info(f"Columns after cleaning names: {df.columns.tolist()}")

        # --- Explicit Column Renaming ---
        # Create a rename dictionary based on the explicit mapping, only including columns that exist in the DataFrame
        rename_dict = {original: standardized for original, standardized in explicit_pv_mapping.items() if original in df.columns}

        # Check if the critical 'Datum und Uhrzeit' and 'PV Leistung' columns were found
        if 'Datum und Uhrzeit' not in df.columns:
             raise ValueError(f"Critical timestamp column 'Datum und Uhrzeit' not found in {filepath}. Found columns: {df.columns.tolist()}")
        if 'PV Leistung' not in df.columns:
             # Log a warning, but don't necessarily raise an error if power is missing,
             # as subsequent steps might handle it (e.g., merging with NaNs)
             logging.error(f"Critical power column 'PV Leistung' not found in {filepath}. PV power data will be missing.")
             # Ensure 'power_w' will be a column later, even if all NaNs, to prevent downstream errors
             # This is handled after selecting columns.

        if rename_dict:
            df = df.rename(columns=rename_dict).copy() # Apply renaming
            logging.info(f"Renamed columns based on explicit mapping: {rename_dict}")
        else:
            logging.warning("No columns matched the explicit PV mapping. Proceeding with original column names (except timestamp).")
            # Ensure timestamp column is still handled if mapping failed for it
            if 'Datum und Uhrzeit' in df.columns:
                 df = df.rename(columns={'Datum und Uhrzeit': 'datetime'}).copy()


        # --- Datetime Parsing and Indexing ---
        # The datetime column should now be named 'datetime'
        if 'datetime' not in df.columns:
             raise ValueError("Timestamp column could not be identified or renamed to 'datetime'. Cannot process PV data.")

        # Strip any whitespace from values in the datetime column before parsing
        if df['datetime'].dtype == 'object':
             df['datetime'] = df['datetime'].str.strip()

        # Parse timestamps with the specified format
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M', errors='coerce')

        # Drop rows where datetime conversion failed
        initial_rows = len(df)
        df.dropna(subset=['datetime'], inplace=True)
        if len(df) < initial_rows:
             logging.warning(f"Dropped {initial_rows - len(df)} rows due to invalid datetime format in PV data after renaming.")


        # Set datetime as index BEFORE localization/conversion
        df.set_index('datetime', inplace=True)

        # --- Timezone Handling ---
        # Explicitly set timezone to local time ('Europe/Vienna') and convert to UTC
        local_tz_name = 'Europe/Vienna'
        try:
            df.index = df.index.tz_localize(local_tz_name, ambiguous='NaT', nonexistent='NaT')
            if df.index.isna().any():
                 num_na_tz = df.index.isna().sum()
                 logging.warning(f"Dropped {num_na_tz} PV rows due to ambiguous/nonexistent timestamps during localization to {local_tz_name}.")
                 df.dropna(inplace=True) # Drop rows that couldn't be localized

            df.index = df.index.tz_convert(self.tz_utc)
            logging.info(f"Converted PV timestamps from local time ({local_tz_name}) to UTC.")
        except Exception as e:
            logging.error(f"Error localizing or converting PV timestamps using '{local_tz_name}': {e}")
            # Fallback or re-raise based on desired strictness
            raise # Re-raise the exception on critical timestamp failure


        # The index is now timezone-aware UTC
        logging.info(f"Processed PV data index timezone: {df.index.tz}")

        # --- Select and Convert Value Columns ---
        # Identify columns that were successfully renamed (excluding the index)
        value_cols_standardized = [col for col in explicit_pv_mapping.values() if col in df.columns and col != 'datetime']

        if not value_cols_standardized:
             logging.warning("No standardized value columns identified after renaming. Returning empty DataFrame for values.")
             return pd.DataFrame(index=df.index) # Return df with just the index

        # Select only the standardized value columns
        df_processed_pv = df[value_cols_standardized].copy()

        # Convert selected value columns to numeric
        for col in df_processed_pv.columns:
             df_processed_pv[col] = pd.to_numeric(df_processed_pv[col], errors='coerce')
             # Log number of NaNs introduced during numeric conversion for these columns
             nan_count_numeric = df_processed_pv[col].isna().sum()
             if nan_count_numeric > 0:
                 logging.warning(f"Converted '{col}' to numeric, found {nan_count_numeric} NaN values.")


        # --- Final Check for 'power_w' ---
        logging.info("--- Debugging Final Processed PV DataFrame ---")
        if 'power_w' in df_processed_pv.columns:
             logging.info("'power_w' column successfully identified, processed, and included.")
             logging.info(f"Final 'power_w' data type: {df_processed_pv['power_w'].dtype}")
             logging.info(f"Sample of final processed PV data (power_w):\n{df_processed_pv['power_w'].head()}")
             logging.info(f"Number of NaNs in 'power_w' after processing: {df_processed_pv['power_w'].isna().sum()}")
        else:
             # This should ideally not happen if 'PV Leistung' was found, but as a safeguard
             logging.error("FATAL: 'power_w' column is NOT present in the final processed PV DataFrame.")
             # Add 'power_w' column filled with NaNs to prevent later errors
             df_processed_pv['power_w'] = np.nan
             logging.warning("Added 'power_w' column filled with NaNs.")

        logging.info("--- End Debugging Final Processed PV DataFrame ---")


        logging.info(f"Processed PV data shape: {df_processed_pv.shape}")
        logging.info("Sample of processed PV data:")
        logging.info(df_processed_pv.head())

        return df_processed_pv


    def save_to_parquet(self, df: pd.DataFrame, output_path: str):
        """Save processed data to parquet format with compression."""
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir: # Check if output_path includes a directory
             os.makedirs(output_dir, exist_ok=True)

        # Convert to PyArrow table and save
        try:
            # Ensure index name is set before saving, otherwise it defaults to '__index_level_0__' in parquet metadata
            if df.index.name is None:
                df.index.name = 'datetime'
            table = pa.Table.from_pandas(df)
            pq.write_table(table, output_path, compression='snappy')
            logging.info(f"Data saved successfully to {output_path}")
        except Exception as e:
            logging.error(f"Error saving data to parquet {output_path}: {e}")


    def plot_data_quality(self, df: pd.DataFrame, title_suffix: str = ""):
        """Plots key variables including AOI to assess data quality."""
        logging.info(f"Generating data quality plots{title_suffix}...")

        # Check if key columns exist before plotting
        required_ts_cols_basic = ['GlobalRadiation [W m-2]', 'power_w']
        required_ts_cols_pvlib = ['ClearSkyGHI', 'AOI [degrees]', 'ClearSkyIndex'] # Pvlib specific
        required_scatter_cols = ['GlobalRadiation [W m-2]', 'power_w']

        # Determine which plots can be generated based on available data
        can_plot_basic_ts = all(col in df.columns for col in required_ts_cols_basic)
        can_plot_pvlib_ts = all(col in df.columns for col in required_ts_cols_pvlib)
        can_plot_scatter = all(col in df.columns for col in required_scatter_cols)

        num_plots = (2 if can_plot_basic_ts else 0) + (1 if can_plot_pvlib_ts else 0) + (1 if can_plot_scatter else 0)

        if num_plots == 0:
             logging.warning("No plots can be generated with available columns.")
             # Print available columns for debugging
             logging.info(f"Available columns for plotting: {df.columns.tolist()}")
             return

        # Dynamically adjust subplot layout
        # sharex=True if any time series plots are being made
        share_x = can_plot_basic_ts or can_plot_pvlib_ts
        fig, axes = plt.subplots(num_plots, 1, figsize=(15, 4.5 * num_plots), sharex=share_x)

        # If only one plot, axes is not an array, handle this case
        if num_plots == 1:
            axes = [axes]

        ax_idx = 0

        if can_plot_basic_ts:
            # Plot Global Radiation and PV Power
            plot_cols = [col for col in required_ts_cols_basic if col in df.columns] # Only plot available columns
            if plot_cols:
                 df[plot_cols].plot(ax=axes[ax_idx])
                 axes[ax_idx].set_title(f'Global Radiation (Measured) vs. PV Power{title_suffix}')
                 axes[ax_idx].set_ylabel('Value') # Generic label as units differ
                 axes[ax_idx].grid(True)
                 ax_idx += 1

            # Plot PV Power separately for clearer view of zero values
            if 'power_w' in df.columns:
                 df['power_w'].plot(ax=axes[ax_idx], color='orange')
                 axes[ax_idx].set_title(f'PV Power Output{title_suffix}')
                 axes[ax_idx].set_ylabel('Power [W]')
                 axes[ax_idx].grid(True)
                 ax_idx += 1


        if can_plot_pvlib_ts:
             # Plot Clear Sky GHI, Angle of Incidence (AOI) and Clear Sky Index
             # Using secondary_y for ClearSkyIndex as its scale is different
             plot_cols = [col for col in required_ts_cols_pvlib if col in df.columns]
             csi_col = 'ClearSkyIndex'
             if csi_col in plot_cols: plot_cols.remove(csi_col) # Remove CSI for first y-axis

             if plot_cols or (csi_col in df.columns): # Plot if at least one required pvlib column or CSI is present
                  plot_df = df[plot_cols + ([csi_col] if csi_col in df.columns else [])].copy() # Include CSI for secondary axis

                  # Fill NaNs for plotting if needed, e.g., with 0 or forward/backward fill
                  # Be careful with filling for plotting vs. training - this fill is just for visualization
                  plot_df_filled = plot_df.fillna(0) # Example: fill with 0 for plotting

                  if plot_cols: # Plot columns on the first y-axis
                       plot_df_filled[plot_cols].plot(ax=axes[ax_idx])
                       axes[ax_idx].set_ylabel('Irradiance [W/m²] / Angle [degrees]') # Combined label
                       axes[ax_idx].grid(True)
                       # Adjust y-limit dynamically based on the first y-axis data
                       y1_max = max(plot_df_filled[col].max() for col in plot_cols) if plot_cols else 0
                       axes[ax_idx].set_ylim(0, y1_max * 1.1 or 180) # Set max to 1.1 * max or 180 if empty/zero

                  else:
                       axes[ax_idx].set_ylabel('') # No label if no columns on primary axis


                  axes[ax_idx].set_title(f'Clear Sky GHI, AOI, and Clear Sky Index{title_suffix}')
                  axes[ax_idx].grid(True)


                  # Plot Clear Sky Index on secondary y-axis if available
                  if csi_col in plot_df_filled.columns:
                       axes[ax_idx].twinx() # Create twin axis even if primary is empty
                       plot_df_filled[csi_col].plot(ax=axes[ax_idx].right_ax, color='red', linestyle='--')
                       axes[ax_idx].right_ax.set_ylabel('Clear Sky Index')
                       axes[ax_idx].right_ax.set_ylim(0, self.CSI_HIGH_THRESHOLD_DAYTIME + 0.5) # Adjust CSI max limit
                  ax_idx += 1
             else:
                  logging.warning(f"Skipping PVlib TS plots due to missing required columns: {required_ts_cols_pvlib}.")


        if can_plot_scatter:
             # Plot Global Radiation vs. PV Power (Scatter) for a subset of data
             # Sample up to 10000 points for performance, or plot all if data is small
             sample_size = min(10000, len(df))
             plot_cols = [col for col in required_scatter_cols if col in df.columns]

             if sample_size > 0 and len(plot_cols) == 2:
                  # Ensure columns are numeric and drop NaNs for scatter plot
                  df_numeric_subset = df[plot_cols].apply(pd.to_numeric, errors='coerce').dropna().sample(min(sample_size, df.dropna(subset=plot_cols).shape[0]), random_state=42)

                  if not df_numeric_subset.empty:
                       sns.scatterplot(x='GlobalRadiation [W m-2]', y='power_w', data=df_numeric_subset, alpha=0.6, s=10, ax=axes[ax_idx])
                       axes[ax_idx].set_title(f'PV Power vs. Global Radiation (Sample, N={len(df_numeric_subset)}){title_suffix}')
                       axes[ax_idx].set_xlabel('Global Radiation [W/m²]')
                       axes[ax_idx].set_ylabel('PV Power [W]')
                       axes[ax_idx].grid(True)
                  else:
                       axes[ax_idx].set_title('Not enough valid numeric data for scatter plot')
                       axes[ax_idx].set_xlabel('Global Radiation [W/m²]')
                       axes[ax_idx].set_ylabel('PV Power [W]')

             else:
                 axes[ax_idx].set_title('Not enough data or missing columns for scatter plot')
                 axes[ax_idx].set_xlabel('Global Radiation [W/m²]')
                 axes[ax_idx].set_ylabel('PV Power [W]')

             ax_idx += 1

        # Set x-label for the last plot if it's a time series plot
        if share_x and num_plots > 0:
             axes[-1].set_xlabel('Time')


        plt.tight_layout()
        plt.show()
        logging.info("Finished generating data quality plots.")

    def plot_specific_variables(self, df: pd.DataFrame, title_suffix: str = ""):
        """
        Plots specific key variables (Power, Global Radiation, AOI, Clear Sky Index)
        as vertical subplots over time.
        """
        logging.info(f"Generating specific variable plots{title_suffix}...")

        # Updated required columns for plotting
        # Note: Temperature is removed, GlobalRadiation and ClearSkyGHI will be on the same plot
        required_cols = ['power_w', 'GlobalRadiation [W m-2]', 'ClearSkyGHI', 'AOI [degrees]', 'ClearSkyIndex']
        available_cols = [col for col in required_cols if col in df.columns]

        # Determine which plots we can actually generate
        # Power plot
        can_plot_power = 'power_w' in available_cols
        # Radiation plot (requires both measured GHI and ClearSkyGHI)
        can_plot_radiation = 'GlobalRadiation [W m-2]' in available_cols and 'ClearSkyGHI' in available_cols
        # AOI plot
        can_plot_aoi = 'AOI [degrees]' in available_cols
        # CSI plot
        can_plot_csi = 'ClearSkyIndex' in available_cols

        # Count the number of subplots needed
        num_subplots = sum([can_plot_power, can_plot_radiation, can_plot_aoi, can_plot_csi])

        if num_subplots == 0:
             logging.warning("No specific variable plots can be generated with available columns.")
             logging.info(f"Available columns: {df.columns.tolist()}")
             return

        # Create vertical subplots sharing the x-axis
        fig, axes = plt.subplots(num_subplots, 1, figsize=(15, 4 * num_subplots), sharex=True)

        # Ensure axes is an array even for 1 subplot
        if num_subplots == 1:
            axes = [axes]

        # Keep track of the current axis index
        ax_idx = 0

        # Plot Power
        if can_plot_power:
            df['power_w'].plot(ax=axes[ax_idx], color='orange', label='PV Power')
            axes[ax_idx].set_title(f'PV Power Output{title_suffix}')
            axes[ax_idx].set_ylabel('Power [W]')
            axes[ax_idx].grid(True)
            axes[ax_idx].legend()
            ax_idx += 1

        # Plot Global Radiation and Clear Sky GHI on the same subplot
        if can_plot_radiation:
            df[['GlobalRadiation [W m-2]', 'ClearSkyGHI']].plot(ax=axes[ax_idx])
            axes[ax_idx].set_title(f'Global Radiation (Measured) vs. Clear Sky GHI{title_suffix}')
            axes[ax_idx].set_ylabel('Irradiance [W/m²]')
            axes[ax_idx].grid(True)
            axes[ax_idx].legend(['Measured GHI', 'Clear Sky GHI']) # Custom legend labels
            ax_idx += 1

        # Plot AOI
        if can_plot_aoi:
            df['AOI [degrees]'].plot(ax=axes[ax_idx], color='deepskyblue', label='AOI')
            axes[ax_idx].set_title(f'Angle of Incidence (AOI){title_suffix}')
            axes[ax_idx].set_ylabel('Angle [degrees]')
            axes[ax_idx].grid(True)
            axes[ax_idx].set_ylim(0, 180) # AOI is typically between 0 and 180 degrees
            axes[ax_idx].legend()
            ax_idx += 1

        # Plot Clear Sky Index
        if can_plot_csi:
            df['ClearSkyIndex'].plot(ax=axes[ax_idx], color='red', label='Clear Sky Index')
            axes[ax_idx].set_title(f'Clear Sky Index{title_suffix}')
            axes[ax_idx].set_ylabel('Clear Sky Index')
            axes[ax_idx].grid(True)
            # Removed: Setting upper limit based on CSI_HIGH_THRESHOLD_DAYTIME
            # axes[ax_idx].set_ylim(0, self.CSI_HIGH_THRESHOLD_DAYTIME + 0.5) # Set limit based on threshold
            axes[ax_idx].legend()
            ax_idx += 1


        # Set the x-label only on the last subplot
        axes[-1].set_xlabel('Time (UTC)')

        plt.tight_layout()
        plt.show()
        logging.info("Finished generating specific variable plots.")


    def process_and_save(self, station_path: str, pv_path: str, high_res_output: str, low_res_output: str):
        """
        Process station data and save both high-resolution and low-resolution datasets,
        ensuring alignment based on the common time range, applying quality filters (removed),
        and calculating solar geometry features.
        """
        # Process station data (10min resolution) - Index becomes UTC-aware
        station_df = self.process_station_data(station_path)

        # Import PV data
        pv_df = self.import_pv_data(pv_path)


        # --- Data Alignment: Clip to common time range ---
        logging.info("\nAligning data to the common time range...")

        if station_df.empty or pv_df.empty:
             logging.error("One of the dataframes is empty after initial processing. Cannot align or merge.")
             # Return empty dataframes with correct index type/timezone if possible
             empty_index_station = station_df.index if not station_df.empty else pd.DatetimeIndex([], tz=self.tz_utc)
             empty_index_pv = pv_df.index if not pv_df.empty else pd.DatetimeIndex([], tz=self.tz_utc)
             combined_index = empty_index_station.union(empty_index_pv) # Get a union of indices to potentially cover the range

             # Create empty dataframes with combined index and correct timezone
             empty_high_res = pd.DataFrame(index=combined_index)
             if empty_high_res.index.tz is None:
                  empty_high_res = empty_high_res.tz_localize(self.tz_utc)

             empty_low_res_index = pd.date_range(start=empty_high_res.index.min().floor(self.low_res_frequency) if not empty_high_res.empty else None,
                                                  end=empty_high_res.index.max().ceil(self.low_res_frequency) if not empty_high_res.empty else None,
                                                  freq=self.low_res_frequency, tz=self.tz_utc)
             empty_low_res = pd.DataFrame(index=empty_low_res_index)

             return empty_high_res, empty_low_res

        # Find the latest start time and earliest end time across both dataframes
        # Both indices are UTC-aware, so max/min works directly
        common_start_time = max(station_df.index.min(), pv_df.index.min())
        common_end_time = min(station_df.index.max(), pv_df.index.max())

        logging.info(f"Station data range: {station_df.index.min()} to {station_df.index.max()}")
        logging.info(f"PV data range: {pv_df.index.min()} to {pv_df.index.max()}")
        logging.info(f"Common time range: {common_start_time} to {common_end_time}")

        # Filter both dataframes to this common time range
        # Indices are UTC-aware, so .loc slicing works correctly
        station_df = station_df.loc[common_start_time:common_end_time].copy()
        pv_df = pv_df.loc[common_start_time:common_end_time].copy()


        logging.info(f"Station data shape after clipping: {station_df.shape}")
        logging.info(f"PV data shape after clipping: {pv_df.shape}")

        if station_df.empty or pv_df.empty:
             logging.error("One of the dataframes is empty after clipping to common range. Cannot merge.")
             # Return empty dataframes with correct index type/timezone
             empty_index_station = station_df.index if not station_df.empty else pd.DatetimeIndex([], tz=self.tz_utc)
             empty_index_pv = pv_df.index if not pv_df.empty else pd.DatetimeIndex([], tz=self.tz_utc)
             combined_index = empty_index_station.union(empty_index_pv) # Get a union of indices to potentially cover the range

             empty_high_res = pd.DataFrame(index=combined_index)
             if empty_high_res.index.tz is None:
                 empty_high_res = empty_high_res.tz_localize(self.tz_utc)

             empty_low_res_index = pd.date_range(start=empty_high_res.index.min().floor(self.low_res_frequency) if not empty_high_res.empty else None,
                                                  end=empty_high_res.index.max().ceil(self.low_res_frequency) if not empty_high_res.empty else None,
                                                  freq=self.low_res_frequency, tz=self.tz_utc)
             empty_low_res = pd.DataFrame(index=empty_low_res_index)


             return empty_high_res, empty_low_res


        # Resample PV data to 10min *after* clipping to match station data frequency
        logging.info("Resampling clipped PV data to 10min frequency...")

        # Debug the frequency of timestamps in both datasets
        station_time_diff = pd.Series(station_df.index[1:]) - pd.Series(station_df.index[:-1])
        logging.info(f"Station data time differences (first 5): {station_time_diff[:5]}")
        if not station_time_diff.empty:
             logging.info(f"Station data mode time diff: {station_time_diff.mode()[0]}")
        else:
             logging.warning("Station data time differences calculation resulted in empty series.")


        # Create a uniform 10-minute frequency index covering the entire period
        # This ensures both dataframes will have exactly matching timestamps
        # Ensure start and end times are not NaT before flooring/ceiling
        if pd.isna(common_start_time) or pd.isna(common_end_time):
            logging.error("Common start or end time is NaT. Cannot create uniform index.")
            # Handle this error - perhaps return empty dataframes or raise exception
            # For now, re-raise the error that caused NaT if possible, or return empty.
            # Assuming the NaT check above handles this, proceed if times are valid.
            pass # If we reached here, common_start/end_time should be valid datetimes

        full_range_index = pd.date_range(
            start=common_start_time.floor(self.high_res_frequency),
            end=common_end_time.ceil(self.high_res_frequency),
            freq=self.high_res_frequency,
            tz=self.tz_utc
        )
        logging.info(f"Created uniform index with {len(full_range_index)} timestamps")

        # Create new dataframes with the uniform index
        # First for station data - use reindex with nearest neighbor to align
        station_df_uniform = pd.DataFrame(index=full_range_index)
        for col in station_df.columns:
            station_df_uniform[col] = station_df[col].reindex(full_range_index, method='nearest')

        logging.info(f"Created uniform station dataframe with shape: {station_df_uniform.shape}")

        # Now for PV data - use reindex with nearest neighbor to align
        if 'power_w' not in pv_df.columns:
            logging.warning("'power_w' not found in PV data after processing. Reindexing PV data will be skipped for power.")
            pv_df_uniform = pd.DataFrame(index=full_range_index) # Create empty with uniform index
        else:
             # Create a new dataframe with the uniform index
             pv_df_uniform = pd.DataFrame(index=full_range_index)

             # Reindex all columns from the original PV dataframe to the uniform index
             # This preserves original values where timestamps match or are very close,
             # and introduces NaNs where there's no close match (which will be filled later).
             for col in pv_df.columns:
                 pv_df_uniform[col] = pv_df[col].reindex(full_range_index, method='nearest')

             # Debug output for raw power values
             logging.info(f"Raw PV power values stats before reindexing: min={pv_df['power_w'].min()}, max={pv_df['power_w'].max()}, non-zero count={(pv_df['power_w'] > 0).sum()}")

             # Debug check
             logging.info(f"PV power values in pv_df_uniform after reindexing: NaN count={pv_df_uniform['power_w'].isna().sum()}")
             # Sample of power values
             logging.info(f"First 5 power values in pv_df_uniform: {pv_df_uniform['power_w'].head().tolist()}")
             # Check non-zero count AFTER reindexing (should be similar to before reindexing)
             logging.info(f"Uniform PV power_w stats in pv_df_uniform after reindexing: min={pv_df_uniform['power_w'].min()}, max={pv_df_uniform['power_w'].max()}, non-zero count={(pv_df_uniform['power_w'] > 0).sum()}")


        # Now simply combine the dataframes using the index directly (since they share the same uniform index)
        logging.info(f"Merging uniform dataframes by direct concatenation...")
        merged_df = pd.concat([station_df_uniform, pv_df_uniform], axis=1)

        logging.info(f"Merged data shape after direct concatenation: {merged_df.shape}")
        logging.info(f"Merged data index timezone: {merged_df.index.tz}")

        # Debug check power values in combined dataframe before filling NaNs
        if 'power_w' in merged_df.columns:
            power_nan_count = merged_df['power_w'].isna().sum()
            power_zeros = (merged_df['power_w'] == 0).sum()
            power_gt_zero = (merged_df['power_w'] > 0).sum()
            logging.info(f"Power stats in merged_df BEFORE final NaN filling: NaN={power_nan_count}, zeros={power_zeros}, >0={power_gt_zero}")

        # --- Debugging: Check NaN count in merged data after merge ---
        logging.info("--- Debugging NaNs in merged data after merge ---")
        if 'power_w' in merged_df.columns:
            nan_count_merged_after_concat = merged_df['power_w'].isna().sum()
            logging.info(f"NaN count in merged_df['power_w'] AFTER concat: {nan_count_merged_after_concat}")
        else:
            logging.warning("'power_w' column not found in merged data after concat.")
        logging.info("--- End Debugging NaNs in merged data after merge ---")
        # --- End Debugging ---

        # --- End Data Alignment ---

        # Calculate solar geometry features (solar position, clear sky, AOI, solar night mask)
        # This function now expects and returns a dataframe with a UTC-aware index
        logging.info("Calculating solar geometry features...")
        # Pass merged_df to the function which will return it with added columns
        merged_df = self.calculate_solar_geometry_features(merged_df)


        # Add irradiation-based night mask
        logging.info("Adding irradiation-based night mask...")
        merged_df = self.add_irradiation_night_mask(merged_df) # Renamed function to be specific

        # Combine night masks if both irradiation ('isNight_irradiation') and solar position ('isNight_solar') masks are available
        if 'isNight_irradiation' in merged_df.columns and 'isNight_solar' in merged_df.columns:
             # Use logical OR - if either method says it's night, consider it night
             merged_df['isNight_combined'] = ((merged_df['isNight_irradiation'] == 1) | (merged_df['isNight_solar'] == 1)).astype(int)
             merged_df = merged_df.drop(['isNight_irradiation', 'isNight_solar'], axis=1) # Drop individual masks
             merged_df = merged_df.rename(columns={'isNight_combined': 'isNight'}) # Rename combined to isNight
             logging.info("Combined irradiation and solar position based night masks into 'isNight'.")
        elif 'isNight_solar' in merged_df.columns:
             merged_df = merged_df.rename(columns={'isNight_solar': 'isNight'}) # Use solar mask if only it exists
             logging.warning("Used solar position based night mask as irradiation-based mask was not available.")
        elif 'isNight_irradiation' in merged_df.columns:
             merged_df = merged_df.rename(columns={'isNight_irradiation': 'isNight'}) # Use irradiation mask if only it exists
             logging.warning("Used irradiation based night mask as solar-based mask was not available.")
        else:
             logging.warning("No 'isNight' mask could be created (missing radiation or solar position data).")
             # Add a default 'isNight' column with 0s if none could be created to prevent errors later
             if 'isNight' not in merged_df.columns: # Only add if it doesn't exist from previous steps
                  merged_df['isNight'] = 0
                  logging.warning("Added a default 'isNight' column with all zeros.")


        # --- Apply Data Quality Filtering (Removed as requested) ---
        logging.info("\nApplying data quality filtering (removed as requested)...")
        # The previous filtering steps (dropping rows based on NaNs, anomalous power, unrealistic CSI) are removed.
        # The dataframe now contains all rows from the common time range, with 'power_w' NaNs filled with 0.0.
        logging.info(f"Shape after removing quality filtering: {merged_df.shape}")

        # --- End Data Quality Filtering ---


        # Handle any remaining missing values in target columns *after* removing filtering
        logging.info("Handling remaining missing values in target columns...")
        # Note: energy_interval might not exist if not in original mapping/data
        # Include 'power_w' here for final NaN filling
        target_cols_high_res = ['power_w', 'energy_wh_symo', 'energy_production_wh'] # Use the standardized names expected to be in merged_df

        for col in target_cols_high_res:
             if col in merged_df.columns:
                  missing_count = merged_df[col].isna().sum()
                  if missing_count > 0:
                       logging.warning(f"Found {missing_count} remaining missing values in '{col}'. Filling with 0.0.")
                       merged_df[col].fillna(0.0, inplace=True) # Fill all remaining NaNs with 0
             else:
                  # This warning should be less frequent with improved PV import and reindexing,
                  # but helpful for debugging if a column vanishes unexpectedly.
                  logging.warning(f"Target column '{col}' not found in merged dataframe for NaN handling.")

        # Debug check power values in combined dataframe after filling NaNs
        if 'power_w' in merged_df.columns:
            power_nan_count_after_fill = merged_df['power_w'].isna().sum()
            power_zeros_after_fill = (merged_df['power_w'] == 0).sum()
            power_gt_zero_after_fill = (merged_df['power_w'] > 0).sum()
            logging.info(f"Power stats in merged_df AFTER final NaN filling: NaN={power_nan_count_after_fill}, zeros={power_zeros_after_fill}, >0={power_gt_zero_after_fill}")


        # Add time-based features to high-resolution data
        # This function expects and returns a dataframe with a timezone-aware index
        logging.info("Adding time-based features to high-resolution data...")
        high_res_df = self.add_time_features(merged_df) # add_time_features returns a copy


        # Save high-resolution dataset (10min)
        # Parquet handles timezone-aware indices correctly
        logging.info(f"Saving high-resolution (10min) dataset...")
        logging.info(f"High-resolution dataset shape: {high_res_df.shape}")
        logging.info("Sample of high-resolution data:")
        logging.info(high_res_df.head())
        logging.info(f"High-resolution data index timezone: {high_res_df.index.tz}")
        self.save_to_parquet(high_res_df, high_res_output)


        # Create low-resolution dataset by resampling to hourly
        logging.info("Creating low-resolution (1h) dataset by resampling...")

        # Define aggregation methods for each column
        agg_dict = {
            'GlobalRadiation [W m-2]': 'mean',
            'Temperature [degree_Celsius]': 'mean',
            'WindSpeed [m s-1]': 'mean',
            'Precipitation [mm]': 'sum', # Sum precipitation over the hour
            'Pressure [hPa]': 'mean',
            'SolarZenith [degrees]': 'mean', # Aggregate solar angles
            'SolarAzimuth [degrees]': 'mean',
            'ClearSkyGHI': 'mean',
            'ClearSkyDNI': 'mean',
            'ClearSkyDHI': 'mean',
            'ClearSkyIndex': 'mean',
            'AOI [degrees]': 'mean', # Aggregate AOI
            'isNight': 'max', # If any 10min period in the hour is night, consider the hour as night
            'energy_wh': 'last', # Cumulative value (take last reading in the hour)
            'energy_interval': 'sum', # Sum up interval energy over the hour (assuming this is calculated or exists)
            'power_w': 'mean' # Average power over the hour
            # Time features will be re-added after resampling
            # 'hour', 'day_of_year', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        }

        # Filter to include only columns that exist in the dataframe
        agg_dict = {k: v for k, v in agg_dict.items() if k in high_res_df.columns}
        if not agg_dict:
             logging.warning("Aggregation dictionary is empty. No columns to aggregate for low-resolution data.")
             # Create an empty dataframe with an hourly index if no columns can be aggregated
             # Resampling an empty df will raise an error, so create the index manually
             if not high_res_df.empty:
                  low_res_index = pd.date_range(start=high_res_df.index.min().floor(self.low_res_frequency),
                                                 end=high_res_df.index.max().ceil(self.low_res_frequency),
                                                 freq=self.low_res_frequency, tz=self.tz_utc) # Maintain UTC timezone
             else:
                  low_res_index = pd.DatetimeIndex([], tz=self.tz_utc)

             low_res_df = pd.DataFrame(index=low_res_index)

        else:
            # Resample to hourly frequency and aggregate
            # Ensure index is sorted before resampling
            if not high_res_df.index.is_monotonic_increasing:
                 high_res_df = high_res_df.sort_index()
                 logging.warning("High-resolution dataframe index was not monotonic, sorting before resampling.")

            # Using origin='start' ensures that the 1h bin ends at XX:00
            # Resampling maintains timezone awareness (UTC)
            low_res_df = high_res_df.resample(self.low_res_frequency, origin='start').agg(agg_dict).copy() # Use .copy()

        logging.info(f"Low-resolution dataframe shape after resampling: {low_res_df.shape}")
        logging.info(f"Low-resolution data index timezone: {low_res_df.index.tz}")


        # Add time-based features to low-resolution data
        # This function expects and returns a dataframe with a timezone-aware index
        logging.info("Adding time-based features to low-resolution data...")
        low_res_df = self.add_time_features(low_res_df) # add_time_features returns a copy

        # Save low-resolution dataset (1h)
        # Parquet handles timezone-aware indices correctly
        logging.info(f"Saving low-resolution (1h) dataset...")
        logging.info(f"Low-resolution dataset shape: {low_res_df.shape}")
        logging.info("Sample of low-resolution data:")
        logging.info(low_res_df.head())
        self.save_to_parquet(low_res_df, low_res_output)

        logging.info("Processing complete!")
        return high_res_df, low_res_df


if __name__ == "__main__":
    # File paths
    # IMPORTANT: Update these paths to your actual file locations
    STATION_PATH = "data/station_data_leoben/Messstationen Zehnminutendaten v2 Datensatz_20220713T0000_20250304T0000.csv"
    PV_PATH = "data/raw_data_evt_act/merge.CSV" # Assuming this is the correct path to your PV data file
    HIGH_RES_OUTPUT = "data/processed/station_data_10min.parquet"
    LOW_RES_OUTPUT = "data/processed/station_data_1h.parquet"

    # Ensure output directory exists
    os.makedirs("data/processed", exist_ok=True)

    # Initialize and run processor
    # Location and PV parameters are now set within the StationDataProcessor __init__
    processor = StationDataProcessor(chunk_size=100000)

    try:
        high_res_df, low_res_df = processor.process_and_save(STATION_PATH, PV_PATH, HIGH_RES_OUTPUT, LOW_RES_OUTPUT)

        # Check if dataframes are not empty before plotting
        if not high_res_df.empty:
            # Generate generic quality plots (optional, can be commented out)
            # processor.plot_data_quality(high_res_df, title_suffix=" (High-Res, After Processing)")

            # Generate the specific requested plots
            processor.plot_specific_variables(high_res_df, title_suffix=" (High-Res, After Processing)")

        # You could also plot low-resolution data if interested
        # if not low_res_df.empty:
        #      processor.plot_specific_variables(low_res_df, title_suffix=" (Low-Res, After Processing)")

    except FileNotFoundError as e:
        logging.error(f"Input file not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}", exc_info=True) # Print traceback
