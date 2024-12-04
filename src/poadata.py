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



def get_pvgis_data(latitude, longitude, start_year, end_year, tilt, azimuth):
    """
    Retrieve hourly plane-of-array (POA) irradiance data from the PVGIS API.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        start_year (int): Start year for the data retrieval.
        end_year (int): End year for the data retrieval.
        tilt (float): Tilt angle of the PV surface in degrees.
        azimuth (float): Azimuth angle of the PV surface in degrees.

    Returns:
        pd.DataFrame: DataFrame containing the POA irradiance data.
    """
    poa_data, meta, inputs = pvlib.iotools.get_pvgis_hourly(
        latitude=latitude, longitude=longitude,
        start=start_year, end=end_year, raddatabase="PVGIS-SARAH2",
        components=True, surface_tilt=tilt, surface_azimuth=azimuth,
        outputformat='json', usehorizon=True, userhorizon=None,
        pvcalculation=False, peakpower=None, pvtechchoice='crystSi',
        mountingplace='free', loss=0, trackingtype=0, optimal_surface_tilt=False,
        optimalangles=False, url='https://re.jrc.ec.europa.eu/api/v5_2/',
        map_variables=True, timeout=30
    )

    # Calculate diffuse and global POA irradiance
    poa_data['poa_diffuse'] = poa_data['poa_sky_diffuse'] + poa_data['poa_ground_diffuse']
    poa_data['poa_global'] = poa_data['poa_diffuse'] + poa_data['poa_direct']

    return poa_data


if __name__ == "__main__":
    latitude = 47.38770748541585
    longitude = 15.094127778561258
    tilt = 30
    azimuth = 149.716  # azimuth for SOUTH (pvlib = 180°, PVGIS = 0°)

    poa_data_2020 = get_pvgis_data(latitude, longitude, 2020, 2020, tilt, azimuth)

    # Save the data as a CSV file
    poa_data_2020.to_csv("poa_data_2020_Leoben_EVT_io.csv")

    print(poa_data_2020)