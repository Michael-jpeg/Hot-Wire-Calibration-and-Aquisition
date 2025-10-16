from PyDAQmx import Task
from PyDAQmx.DAQmxConstants import *
from PyDAQmx.DAQmxFunctions import *
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ctypes
import os


#output file
date_str = "Insert Date Calibrated Here"
trial_temp = "Insert Calibration Temperature Here"
filename = f"CTA_Calibration_Data_{date_str}_{trial_temp}.xlsx"


# Inputs
sample_rate = 10000.0        # Fixed sample rate (Hz)
num_samples = 50000          # Samples per trial
channel = "Daq"
fan_speeds = [1,2,3,4,5,6,7,8,9,10] #This should a 1 to n list wheren is the desired number of calibration points

# Initial Coefficient Guesses (no need to change)
A = 1
B = .5
n = 0.5

# Storage
raw_data_dict = {}     
avg_data_dict = {}      


# Data Collection Loop
for speed in fan_speeds:
    input(f"\n Run {speed}. Press ENTER when ready to collect data...")
    print(f" Collecting data for point: {speed}")
    
    # Setup DAQ task
    task = Task()
    task.CreateAIVoltageChan(   # Set up DAQ channel to be used
        channel,                # channel to use
        "",                     # desired name for channel
        DAQmx_Val_Cfg_Default,  # terminal mode
        0.0,                    # min voltage
        5.0,                    # max voltage
        DAQmx_Val_Volts,        # units
        None                    # custom scale
    )
    
    task.CfgSampClkTiming(      # Set up sample clock
        "",                     # Clock Source
        sample_rate,            # Sampling rate
        DAQmx_Val_Rising,       # Clock edge
        DAQmx_Val_FiniteSamps,  # Acquisition mode
        num_samples             # Number of samples
   )

    # Read data
    data = np.zeros((num_samples,), dtype=np.float64)
    read = int32()
    
    task.StartTask()
    task.ReadAnalogF64(             # Start Reading Values
       num_samples,                # Samples
       10.0,                       # Wait before sampling
       DAQmx_Val_GroupByChannel,   # Grouping Data
       data,                       # Where to put data
       num_samples,                # Size of data
       byref(read),                # How many samples were read
       None
   )
    task.StopTask()
    task.ClearTask()

    # Store data and average
    raw_data_dict[speed] = data
    avg_data_dict[speed] = np.mean(data)

    # Plot each run (optional)
    

plt.show()

# Convert to DataFrames
raw_data_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in raw_data_dict.items()]))
avg_data_df = pd.DataFrame(avg_data_dict, index=["Average Voltage"])
avg_voltages = np.array(list(avg_data_dict.values()))


######################################
#    Input Fan Speeds                #
######################################

probe_input = input("Enter Flow Speeds (1,2,3,...):")
flow_speeds = np.array([float(x.strip())for x in probe_input.split(',')])

avg_data_df.columns = flow_speeds

    
######################################
#    Find Kings Law Constants        #
######################################


R_squared = 0
E_squared_measured = [i**2 for i in avg_voltages]


def power(list):
    return [i**n for i in list]
    
# Compute predicted E^2 using King's Law
E_squared_predicted = A + B * flow_speeds**n
    
# Calculate R^2
ss_res = np.sum((E_squared_measured - E_squared_predicted)**2)
ss_tot = np.sum((E_squared_measured - np.mean(E_squared_measured))**2)
R_squared = 1 - (ss_res / ss_tot)
    
    
        
def fit_func(x,A_,B_,n_):
    return A_+ B_*x**n_

initial = [A,B,n]
params, covariance = curve_fit(fit_func, flow_speeds, E_squared_measured,p0 = initial,maxfev=5000)
A,B,n = params
coefs = pd.DataFrame(params)
print(f"Fit Parameters: A= {A:.4f}, B= {B:.4f}, n= {n:.4f}")


######################################
#   Plot Mean Velocity vs.Fan Speed and fit line #
######################################

plt.figure()
plt.plot(flow_speeds, E_squared_measured, label = 'measured')
plt.title(f"Voltage^2 vs Fan Speed ")
plt.xlabel("Speed")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.tight_layout()

x_line = np.linspace(0, flow_speeds[-1], num=1000)
y_line = fit_func(x_line,A,B,n)
plt.plot(x_line,y_line,label = 'fit line')
plt.legend()

plt.show()

# --------------------------
# Export to Excel
# --------------------------
A_round = round(A,2)
B_round = round(B,2)
n_round = round(n,2)
filename = f"CTA_Calibration_Data_{date_str}_{trial_temp}_A{A_round}_B{B_round}_n{n_round}.xlsx"

path = os.path.join("Calibration Books", filename)

# Save both to Excel
with pd.ExcelWriter(path, engine="openpyxl") as writer:
    raw_data_df.to_excel(writer, sheet_name="Raw_Voltages", index=False)
    avg_data_df.to_excel(writer, sheet_name="Averages", index=True)

print(f"\nSaved to '{filename}'")

path = os.path.join("csvs", f"{date_str}_coefs.csv")
coefs.to_csv(path, index=False)


