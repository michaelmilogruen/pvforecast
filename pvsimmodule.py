# -*- coding: utf-8 -*-
"""
Author: Michael Grün
Email: michaelgruen@hotmail.com
Description: This script calculates the power output of a photovoltaic system
             using the PVLib library and processes the results in an Excel file.
Version: 1.0
Date: 2024-05-03
"""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import openpyxl
import pandas as pd
import pvlib
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.dimensions import ColumnDimension
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import poadata
import xlwings as xw
from math import sin, cos, pi
from datetime import datetime
from datetime import timedelta
import sys

# Define parameters
start = '2020-01-01 00:00'
end = '2020-12-31 23:00'
latitude = 47.38770748541585
longitude = 15.094127778561258
tilt = 30
azimuth = 149.716  # azimuth for SOUTH (pvlib = 180°, PVGIS = 0°)
celltype = 'polycristalline'
pdc0 = 240  # Nominal max. power in [W] (=Pmp)
v_mp = 29.87  # Voltage at MP [V]
i_mp = 8.04  # Current at MP [A]
v_oc = 37.33  # Open-circuit voltage [V]
i_sc = 8.78  # Short-circuit current [A]
alpha_sc = 0.0041  # Temperature coefficient for i_sc [A/K]
beta_voc = -0.114  # Temperature coefficient for v_oc [V/K]
gamma_pdc = -0.405  # Temperature coefficient for pdc0 (Pmp) [%/K] (directly take procentual value --> V11)
cells_in_series = 3 * 23
temp_ref = 25  # Reference temperature [°C]

# Define some helper functions
def get_weekday(date):
    return date.weekday() + 1  # Excel weekday starts from 1

def get_hour(date):
    return date.hour

def get_month(date):
    return date.month

def execute_vba_actions(file_name: str) -> None:
    """
    Executes various VBA actions on an Excel file using the openpyxl library.

    Args:
        file_name: The name of the Excel file to be processed.

    Returns:
        None. The function modifies the specified Excel file in-place.
    """
    workbook = openpyxl.load_workbook(file_name)
    model_chain_results = workbook['Model Chain Results']
    poa_data = workbook['POA Data']

    # Autofit column F in 'Model Chain Results' worksheet
    model_chain_results.column_dimensions['F'].auto_size = True

    # Autofill range B8712:B8785 in 'Model Chain Results' worksheet
    start_value = model_chain_results['B8712'].value
    for row in range(8712, 8786):
        model_chain_results.cell(row=row, column=2, value=start_value)

    # Autofit column F again
    model_chain_results.column_dimensions['F'].auto_size = True

    # Autofit all columns in 'Model Chain Results' worksheet
    for column in model_chain_results.columns:
        max_length = max(len(str(cell.value)) for cell in column)
        adjusted_width = (max_length + 2) * 1.2
        column_letter = get_column_letter(column[0].column)
        model_chain_results.column_dimensions[column_letter].width = adjusted_width

    # Autofit columns in 'POA Data' worksheet
    for column in poa_data.columns:
        max_length = max(len(str(cell.value)) for cell in column)
        adjusted_width = (max_length + 2) * 1.2
        column_letter = get_column_letter(column[0].column)
        poa_data.column_dimensions[column_letter].width = adjusted_width

    # Copy range from 'POA Data' to 'Model Chain Results'
    poa_data_range = poa_data['B1:J' + str(poa_data.max_row)]
    model_chain_results['G1'].value = None
    for row in poa_data_range:
        for cell in row:
            model_chain_results.cell(row=cell.row, column=cell.column - 1 + 7, value=cell.value)

    # Autofit column G in 'Model Chain Results'
    model_chain_results.column_dimensions['G'].auto_size = True

    worksheet = workbook['Model Chain Results']

    # Define the start cell and date
    start_cell = "A2"
    start_date = datetime(2020, 1, 1)  # Adjust the start date as needed
    row = worksheet[start_cell].row

    # Iterate over the rows
    start_row = worksheet[start_cell].row
    for row_cells in worksheet.iter_rows(min_row=start_row, min_col=1):
        # Inside the loop
        date = start_date + timedelta(days=(row_cells[1].row - row))


        # Calculate weekday
        worksheet.cell(row=row_cells[1].row, column=17, value="Weekday")
        worksheet.cell(row=row_cells[1].row, column=18, value=get_weekday(date))

        # Calculate sin and cos of day
        worksheet.cell(row=row_cells[1].row, column=19, value=sin(get_weekday(date) / 7 * 2 * pi))
        worksheet.cell(row=row_cells[1].row, column=20, value=cos(get_weekday(date) / 7 * 2 * pi))

        # Calculate hour
        worksheet.cell(row=row_cells[1].row, column=21, value="hour")
        worksheet.cell(row=row_cells[1].row, column=22, value=get_hour(date))

        # Calculate sin and cos of hour
        worksheet.cell(row=row_cells[1].row, column=23, value=sin(get_hour(date) / 24 * 2 * pi))
        worksheet.cell(row=row_cells[1].row, column=24, value=cos(get_hour(date) / 24 * 2 * pi))

        # Calculate month
        worksheet.cell(row=row_cells[1].row, column=25, value="month")
        worksheet.cell(row=row_cells[1].row, column=26, value=get_month(date))

        # Calculate sin and cos of month
        worksheet.cell(row=row_cells[1].row, column=27, value=sin(get_month(date) / 12 * 2 * pi))
        worksheet.cell(row=row_cells[1].row, column=28, value=cos(get_month(date) / 12 * 2 * pi))

    # Save the modified workbook
    workbook.save(file_name)



def calculate_power_output(start: str, end: str, latitude: float, longitude: float,
                           tilt: float, azimuth: float, celltype: str, pdc0: int,
                           v_mp: float, i_mp: float, v_oc: float, i_sc: float,
                           alpha_sc: float, beta_voc: float, gamma_pdc: float,
                           cells_in_series: int, temp_ref: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function calculates the power output of the photovoltaic system.

    Args:
        start (str): Start date and time of the simulation in 'YYYY-MM-DD HH:MM' format.
        end (str): End date and time of the simulation in 'YYYY-MM-DD HH:MM' format.
        latitude (float): Latitude of the location in decimal degrees.
        longitude (float): Longitude of the location in decimal degrees.
        tilt (float): Tilt angle of the PV system surface in degrees.
        azimuth (float): Azimuth angle of the PV system surface in degrees.
        celltype (str): Type of PV cell.
        pdc0 (int): Nominal max. power in [W] (=Pmp).
        v_mp (float): Voltage at MP [V].
        i_mp (float): Current at MP [A].
        v_oc (float): Open-circuit voltage [V].
        i_sc (float): Short-circuit current [A].
        alpha_sc (float): Temperature coefficient for i_sc [A/K].
        beta_voc (float): Temperature coefficient for v_oc [V/K].
        gamma_pdc (float): Temperature coefficient for pdc0 (Pmp) [%/K].
        cells_in_series (int): Number of cells in series.
        temp_ref (int): Reference temperature [°C].

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the ac_results and poa_data_2020 DataFrames.
    """
    # Assuming that the PV system is located in Leoben, EVT
    location = Location(latitude=latitude, longitude=longitude,
                        tz='Europe/Vienna', altitude=547.6, name='EVT')

    # Get POA data from the PVGIS API using the iotools call
    poa_data_2020 = poadata.get_pvgis_data(latitude, longitude, 2020, 2020, tilt, azimuth)

    # Save the data as a CSV file
    poa_data_2020.to_csv("poa_data_2020_Leoben_EVT_io.csv")
    poa_data_2020 = pd.read_csv('poa_data_2020_Leoben_EVT_io.csv', index_col=0)
    poa_data_2020.index = pd.date_range(start='2020-01-01 00:00',
                                        periods=len(poa_data_2020.index),
                                        freq="h")
    poa_data = poa_data_2020[start:end]

    # Calculate solar position
    solarpos = location.get_solarposition(times=pd.date_range(start, end=end,
                                                              freq="h"))

    # Calculate angle of incidence and incidence angle modifier
    aoi = pvlib.irradiance.aoi(
        tilt, azimuth, solarpos.apparent_zenith, solarpos.azimuth)
    iam = pvlib.iam.ashrae(aoi)

    # Calculate effective irradiance
    effective_irradiance = poa_data["poa_direct"] + iam + poa_data["poa_diffuse"]

    # Calculate cell temperature
    temp_cell = pvlib.temperature.faiman(poa_data["poa_global"], poa_data["temp_air"], poa_data["wind_speed"])

    # Calculating characteristic single-diode-model-output parameters
    I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = pvlib.ivtools.sdm.fit_cec_sam(celltype=celltype,
                                                                                   v_mp=v_mp,
                                                                                   i_mp=i_mp,
                                                                                   v_oc=v_oc,
                                                                                   i_sc=i_sc,
                                                                                   alpha_sc=alpha_sc,
                                                                                   beta_voc=beta_voc,
                                                                                   gamma_pmp=gamma_pdc,
                                                                                   cells_in_series=cells_in_series)

    # Calculating CEC parameters
    cec_params = pvlib.pvsystem.calcparams_cec(effective_irradiance,
                                               temp_cell,
                                               alpha_sc,
                                               a_ref,
                                               I_L_ref,
                                               I_o_ref,
                                               R_sh_ref,
                                               R_s,
                                               Adjust)

    # 1) Maximum power point
    mpp = pvlib.pvsystem.max_power_point(*cec_params, method="newton")  # Use the cec_params for a single module

    # Setup a new system for handling a PV system, not only a single module
    system = PVSystem(modules_per_string=23, strings_per_inverter=3)

    # Scale the mpp result to this system
    dc_scaled = system.scale_voltage_current_power(mpp)

    # AC output of the system
    cec_inverters = pvlib.pvsystem.retrieve_sam('CECInverter')  # Retrieving database for inverter data
    inverter = cec_inverters['Advanced_Energy_Industries__AE_3TL_23_10_08__480V_']  # 20kW

    ac_results = pvlib.inverter.sandia(
        v_dc=dc_scaled.v_mp,
        p_dc=dc_scaled.p_mp,
        inverter=inverter)

    # Preparing the results dataframe
    results_spec_sheet_df = pd.concat([ac_results, dc_scaled.i_mp, dc_scaled.v_mp, dc_scaled.p_mp, temp_cell], axis=1)

    # Setting custom column names
    results_spec_sheet_df.columns = ['AC Power', 'DC scaled I_mp', 'DC scaled V_mp', 'DC scaled P_mp',
                                     'Cell Temperature']

    # Creating an Excel writer object
    with pd.ExcelWriter("results.xlsx") as writer:
        # Saving the model chain results to the first worksheet
        results_spec_sheet_df.to_excel(writer, sheet_name='Model Chain Results')

        # Saving the original poa data to a new worksheet
        poa_data_2020.to_excel(writer, sheet_name='POA Data')

    return ac_results, poa_data_2020


def plot_results(ac_results: pd.DataFrame, figsize=(16, 9)):
    """
    This function plots the results of the power output calculations.

    Args:
        ac_results (pd.DataFrame): The DataFrame containing the AC power results.
        figsize (tuple, optional): The figure size. Defaults to (16, 9).
    """
    try:
        # Plotting the results - Energy yield from start to end
        ac_results.plot(figsize=figsize)
        plt.title("AC Power - PVSystem")
        plt.xlabel("Time")
        plt.ylabel("Energy Yield")
        plt.grid(True)
        plt.savefig("energy_yield_start_to_end.png")

        # Plotting the results - Energy yield over one year - monthly sum
        monthly_sum = ac_results.resample('ME').sum()
        monthly_sum.plot(figsize=figsize)
        plt.title("AC Power - PVSystem (Monthly Sum)")
        plt.xlabel("Time")
        plt.ylabel("Energy Yield")
        plt.grid(True)
        plt.savefig("energy_yield_monthly_sum.png")

        plt.show()
    except (TypeError, ValueError) as e:
        print(f"An error occurred: {e}")


def main():
    """
    Main function to run the script.
    """
    # Call the separate functions here
    ac_results, poa_data_2020 = calculate_power_output(start, end, latitude, longitude,
                                                       tilt, azimuth, celltype, pdc0,
                                                       v_mp, i_mp, v_oc, i_sc,
                                                       alpha_sc, beta_voc, gamma_pdc,
                                                       cells_in_series, temp_ref)
    plot_results(ac_results)
    execute_vba_actions('results.xlsx')

    return 0


if __name__ == '__main__':
    sys.exit(main())
