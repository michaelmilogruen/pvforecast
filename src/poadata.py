#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Michael Gruen
Email: michaelgruen@hotmail.com
Created: 01.05.2024 08:17
Project: pvforecast
"""

import pandas as pd
import pvlib
import os



def get_pvgis_data(latitude, longitude, start_year, end_year, tilt, azimuth,
                  pvcalculation=False, peakpower=None, pvtechchoice='crystSi',
                  mountingplace='free', loss=0, trackingtype=0):
    """
    Retrieve hourly plane-of-array (POA) irradiance data and optionally PV power output data from the PVGIS API.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        start_year (int): Start year for the data retrieval.
        end_year (int): End year for the data retrieval.
        tilt (float): Tilt angle of the PV surface in degrees.
        azimuth (float): Azimuth angle of the PV surface in degrees.
        pvcalculation (bool, default: False): If True, include PV power output calculation.
        peakpower (float, optional): Nominal power of the PV system in kW. Required if pvcalculation=True.
        pvtechchoice (str, default: 'crystSi'): PV technology. Options: 'crystSi', 'CIS', 'CdTe', 'Unknown'.
        mountingplace (str, default: 'free'): Type of mounting. Options: 'free' for free-standing, 'building' for building-integrated.
        loss (float, default: 0): Sum of PV system losses in percent.
        trackingtype (int, default: 0): Type of tracking. 0=fixed, 1=single horizontal axis aligned north-south,
                                       2=two-axis tracking, 3=vertical axis tracking,
                                       4=single horizontal axis aligned east-west,
                                       5=single inclined axis aligned north-south.

    Returns:
        pd.DataFrame: DataFrame containing the POA irradiance data and, if pvcalculation=True, PV power output.
    """
    # Validate parameters
    if pvcalculation and peakpower is None:
        raise ValueError("peakpower must be provided when pvcalculation is enabled")
    
    if loss < 0 or loss > 100:
        raise ValueError("loss must be between 0 and 100 percent")
    
    # Valid tracking types
    valid_tracking_types = [0, 1, 2, 3, 4, 5]
    if trackingtype not in valid_tracking_types:
        raise ValueError(f"trackingtype must be one of {valid_tracking_types}")
    
    # Valid PV technologies
    valid_pv_techs = ['crystSi', 'CIS', 'CdTe', 'Unknown']
    if pvtechchoice not in valid_pv_techs:
        raise ValueError(f"pvtechchoice must be one of {valid_pv_techs}")
    
    # Valid mounting places
    valid_mounting_places = ['free', 'building']
    if mountingplace not in valid_mounting_places:
        raise ValueError(f"mountingplace must be one of {valid_mounting_places}")
    
    poa_data, meta, inputs = pvlib.iotools.get_pvgis_hourly(
        latitude=latitude, longitude=longitude,
        start=start_year, end=end_year, raddatabase="PVGIS-SARAH3",
        components=True, surface_tilt=tilt, surface_azimuth=azimuth,
        outputformat='json', usehorizon=True, userhorizon=None,
        pvcalculation=pvcalculation, peakpower=peakpower, pvtechchoice=pvtechchoice,
        mountingplace=mountingplace, loss=loss, trackingtype=trackingtype,
        optimal_surface_tilt=False, optimalangles=False,
        url='https://re.jrc.ec.europa.eu/api/v5_3/',
        map_variables=True, timeout=30
    )

    # Calculate diffuse and global POA irradiance
    poa_data['poa_diffuse'] = poa_data['poa_sky_diffuse'] + poa_data['poa_ground_diffuse']
    poa_data['poa_global'] = poa_data['poa_diffuse'] + poa_data['poa_direct']
    
    # Convert timestamps from UTC+0 to UTC+1 (Vienna)
    # Check if 'time' is a column or if it's the index
    if 'time' in poa_data.columns:
        # If 'time' is a column, convert that column
        if pd.api.types.is_datetime64_any_dtype(poa_data['time']):
            if poa_data['time'].dt.tz is None:
                poa_data['time'] = poa_data['time'].dt.tz_localize('UTC')
            poa_data['time'] = poa_data['time'].dt.tz_convert('Europe/Vienna')
    else:
        # If 'time' is the index, convert the index
        if poa_data.index.tz is None:
            poa_data.index = poa_data.index.tz_localize('UTC')
        poa_data.index = poa_data.index.tz_convert('Europe/Vienna')
    
    return poa_data


if __name__ == "__main__":
    latitude = 47.38770748541585
    longitude = 15.094127778561258
    tilt = 30
    azimuth = 149.716  # azimuth for SOUTH (pvlib = 180°, PVGIS = 0°)
    
    # Get data without PV calculation
    poa_data = get_pvgis_data(latitude, longitude, 2022, 2023, tilt, azimuth)
    
    # Example with PV calculation enabled
    poa_data_with_pv = get_pvgis_data(
        latitude, longitude, 2022, 2023, tilt, azimuth,
        pvcalculation=True, peakpower=10.56, loss=14.0
    )
    print("\nSample data with PV calculation:")
    if 'P' in poa_data_with_pv.columns:
        print(f"PV Power output column found: {poa_data_with_pv['P'].head()}")
    else:
        print("Warning: No PV power output column found in the results")

    # Print the timezone information after conversion
    if 'time' in poa_data.columns:
        print(f"Timestamp timezone: {poa_data['time'].dt.tz}")
    else:
        print(f"Timestamp timezone: {poa_data.index.tz}")

    # Save the data as a CSV file with Vienna timezone indicator
    # Get the script directory and construct proper paths relative to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "data", "pvgis_data")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the regular POA data
    output_path = os.path.join(output_dir, "poa_data.csv")
    poa_data.to_csv(output_path)
    print(f"POA irradiance data saved to: {output_path}")
    
    # Save the PV calculation data to a separate file
    pv_output_path = os.path.join(output_dir, "poa_data_with_pv.csv")
    poa_data_with_pv.to_csv(pv_output_path)
    print(f"PV calculation data saved to: {pv_output_path}")

    print(poa_data.head())