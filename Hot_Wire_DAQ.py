from PyDAQmx import Task
from PyDAQmx.DAQmxConstants import *
from PyDAQmx.DAQmxFunctions import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import os

# Inputs
sample_rate = 50000        # Fixed sample rate (Hz)
num_samples = 500000          # Samples per trial


channel = "DAQ channel name"
filename = "csv file with Kings Law Coefficients"
savename = 'Data output file'


coefs = pd.read_csv(f"csvs/{filename}")
A,B,n = coefs['0'].values


# Storage
raw_data = np.array([])

# Data Collection Loop

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

voltages = data
low = A**(1/2)+0.00000001
voltages = np.array([low if x < low else x for x in voltages])
velocities = ((voltages**2-A)/B)**(1/n)

time_axis = np.linspace(0, (num_samples - 1) / sample_rate, num_samples)
plt.figure()
plt.plot(time_axis, velocities)
plt.title(f"Velocity vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (V)")
plt.grid(True)
plt.tight_layout()

plt.show()



path = os.path.join("Turbulence_Data/tone_invstgtn", savename)
Hot_Wire_Data = pd.DataFrame({'Voltages': voltages, 'Velocities': velocities})
Hot_Wire_Data.to_csv(path, index=False)
print(f'{savename} saved')