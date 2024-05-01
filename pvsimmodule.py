"""
Author: Michael Grün
Email: michaelgruen@hotmail.com
Description: This script calculates the power output of a photovoltaic system
             using the PVLib library and processes the results in an Excel file.
Version: 1.0
Date: 2023-05-15
"""

import pvlib
from pvlib.modelchain import ModelChain
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import os
import xlwings as xw
import matplotlib.pyplot as plt
import pandas as pd

def edit_excel(excel_file):
    """
    This function edits an Excel workbook by running a VBA macro.
    The macro performs the following tasks:
    1. Creates a new worksheet named "RESULTS" (if it doesn't exist)
    2. Copies the timestamp column from the "Model Chain Results" sheet to the "RESULTS" sheet
    3. Inserts the weekday, month, and hour columns in the "RESULTS" sheet
    4. Copies the "Power AC" column from the "Model Chain Results" sheet to the "RESULTS" sheet
    5. Copies the temperature column from the "TMY Data" sheet to the "RESULTS" sheet

    Args:
        excel_file (str): The path to the Excel file.
    """
    # Open the specified Excel workbook
    wb = xw.Book(excel_file)

    # VBA code as a Python string
    vba_code = """
    'CREATE WORKSHEET RESULTS
        Dim ws As Worksheet

        ' Check if the worksheet already exists
        On Error Resume Next
        Set ws = ThisWorkbook.Sheets("RESULTS")
        On Error GoTo 0
    
        ' If the worksheet does not exist, create it as the first sheet
        If ws Is Nothing Then
            Set ws = ThisWorkbook.Sheets.Add(Before:=ThisWorkbook.Sheets(1))
            ws.Name = "RESULTS"
        Else
            MsgBox "Sheet 'RESULTS' already exists."
        End If
        
        '______________________________________________________________________
        
        'COPY TIMESTEMP FROM MODEL CHAIN RESULTS
        
        Dim sourceWs As Worksheet, targetWs As Worksheet
        Dim lastRow As Long
    
        ' Set the source and target worksheets
        Set sourceWs = ThisWorkbook.Sheets("Model Chain Results")
        Set targetWs = ThisWorkbook.Sheets("RESULTS")
    
        ' Find the last row in column A in the source worksheet
        lastRow = sourceWs.Cells(sourceWs.Rows.Count, "A").End(xlUp).Row
    
        ' Copy the entire column A from source to target worksheet
        sourceWs.Range("A1:A" & lastRow).Copy Destination:=targetWs.Range("A1")
        '______________________________________________________________________
        
        'INSERT DAY MONTH AND HOUR
                
        Dim i As Long
        Dim dateTime As Variant
    
        ' Set the worksheet
        Set ws = ThisWorkbook.Sheets("RESULTS")
    
        ' Find the last row in column A
        lastRow = ws.Cells(ws.Rows.Count, "A").End(xlUp).Row
    
        ' Loop through each row and calculate the weekday, month and hour
        For i = 1 To lastRow
            ' Check if the cell is not empty
            If ws.Cells(i, "A").Value <> "" Then
                ' Extract the date and time value from the timestamp
                dateTime = ws.Cells(i, "A").Value
                
                ' Write the weekday to column B
                ws.Cells(i, "B").Value = WeekdayName(Weekday(dateTime, vbMonday))
                ' Write the month to column C
                ws.Cells(i, "C").Value = MonthName(Month(dateTime))
                ' Write the hour to column D
                ws.Cells(i, "D").Value = Hour(dateTime)
            End If
        Next i
        '______________________________________________________________________
        
        'COPY POWER AC FROM MODEL CHAIN RESULTS
        


        ' Set the source and target worksheets
        Set sourceWs = ThisWorkbook.Sheets("Model Chain Results")
        Set targetWs = ThisWorkbook.Sheets("RESULTS")
    
        ' Find the last row with data in column B on the source worksheet
        lastRow = sourceWs.Cells(sourceWs.Rows.Count, "B").End(xlUp).Row
    
        ' Loop through each row and copy data from source to target column E
        For i = 1 To lastRow
            ' If source cell is empty, write 0, otherwise copy the value
            If IsEmpty(sourceWs.Cells(i, "B").Value) Then
                targetWs.Cells(i, "E").Value = 0
            Else
                targetWs.Cells(i, "E").Value = sourceWs.Cells(i, "B").Value
            End If
        Next i  
        
        '______________________________________________________________________
        
        'COPY TEMPERATURE FROM TMY DATA
        
        Dim wsSource As Worksheet
        Dim wsDest As Worksheet
    
        ' Define worksheets
        Set wsSource = ThisWorkbook.Sheets("TMY Data")
        Set wsDest = ThisWorkbook.Sheets("RESULTS")
    
        ' Copy column B from TMY Data to column F in RESULTS
        wsSource.Columns("B:B").Copy Destination:=wsDest.Columns("F:F")
         
        
    End Sub

    """

    # Add the VBA code to the workbook's VBProject
    wb.api.VBProject.VBComponents.Add(1).CodeModule.AddFromString(vba_code)

    # Run the VBA macro
    wb.macro('FillWeekdayMonthHour')()

    # Save the workbook if needed
    # You can choose to overwrite the existing file or save as a new file
    wb.save(excel_file)  # This will overwrite the existing file
    # wb.save('new_file_path.xlsx')  # This will save as a new file

    # Close the workbook
    wb.close()

# Specifications from the data sheet: KPV_Datenblatt_PE_NEC_Power60_DE_NEU_20120315.pdf
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

# Assuming that the PV system is located in Leoben, EVT
location = Location(latitude=47.38770748541585, longitude=15.094127778561258,
                    tz='Europe/Vienna', altitude=547.6, name='EVT')

surface_tilt = 30
surface_azimuth = 149.716

# Time range for the simulation
# 1) One week - 1st week in July 2020
# start = '2020-07-01 00:00'
# end = '2020-07-07 23:59'

# 2) One month - July 2020
# start = '2020-07-01 00:00'
# end = '2020-07-31 23:00'

# 3) Year 2020
start = '2020-01-01 00:00'
end = '2020-12-31 23:00'

# Get POA data from the PVGIS API using the iotools call
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
    surface_tilt, surface_azimuth, solarpos.apparent_zenith, solarpos.azimuth)
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
mpp.plot(figsize=(16, 9))
plt.grid()
plt.title("MPP - single module")
plt.show()

# Setup a new system for handling a PV system, not only a single module
system = PVSystem(modules_per_string=23, strings_per_inverter=3)

# Scale the mpp result to this system
dc_scaled = system.scale_voltage_current_power(mpp)

dc_scaled.plot(figsize=(16, 9))
plt.grid()
plt.title("DC Power - PVSystem")
plt.show()

# AC output of the system
cec_inverters = pvlib.pvsystem.retrieve_sam('CECInverter')  # Retrieving database for inverter data
inverter = cec_inverters['Advanced_Energy_Industries__AE_3TL_23_10_08__480V_']  # 20kW

ac_results = pvlib.inverter.sandia(
    v_dc=dc_scaled.v_mp,
    p_dc=dc_scaled.p_mp,
    inverter=inverter)

cec_inverters.to_excel("CEC_Inverters_1.xlsx")

# Preparing the results dataframe
results_spec_sheet_df = pd.concat([ac_results, dc_scaled.i_mp, dc_scaled.v_mp, dc_scaled.p_mp, temp_cell], axis=1)

# Setting custom column names
results_spec_sheet_df.columns = ['AC Power', 'DC scaled I_mp', 'DC scaled V_mp', 'DC scaled P_mp', 'Cell Temperature']

# Creating an Excel writer object
with pd.ExcelWriter("complete_spec_sheet_results_poa_data.xlsx") as writer:
    # Saving the model chain results to the first worksheet
    results_spec_sheet_df.to_excel(writer, sheet_name='Model Chain Results')

    # Saving the original poa data to a new worksheet
    poa_data_2020.to_excel(writer, sheet_name='POA Data')

# Plotting the results - 1) Energy yield from start to end (see above for setting of start and end)
ac_results.plot(figsize=(16, 9))
plt.title("AC Power - PVSystem")
plt.plot()
plt.grid()
plt.show()

# # Plotting the results - 2) Energy yield over one year - monthly sum -> trendline
# ac_results.resample('M').sum().plot(figsize=(16, 9))
# plt.title("Leoben_EVT - POA_Data - Monthly Sum")
# # Adding a grid to the plot
# plt.grid()
# plt.show()

