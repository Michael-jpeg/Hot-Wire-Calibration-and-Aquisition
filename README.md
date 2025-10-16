# Hot-Wire-Calibration-and-Aquisition

This repository contains the .py files necessary for Hot Wire Anemometry. This inclued the calibration, data aquistion, and the data processing. The last of which will be added to the repository soon.

## Dependencies 
The following packages must be installed for the code to run

from PyDAQmx import Task
from PyDAQmx.DAQmxConstants import *
from PyDAQmx.DAQmxFunctions import *
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ctypes
import os
import scipy.signal
from scipy.signal import detrend

* Note: The `PyDAQmx` library interfaces with National Instruments hardware, which requires the **NI-DAQmx drivers** to communicate with your DAQ device.  
> Download the drivers from the official NI website: [NI DAQmx Driver Downloads](https://www.ni.com/en-us/support/downloads/drivers/download.ni-daqmx.html)

## There are 2 main files included in the repository
1) Hot_Wire_Calibration.py will take data from a desired number of points and use them to find the coefficients for Kings Law, E^2=A+B*U^(1/n), and produces a CSV of the coefficients
2) Hot_Wire_DAQ.py takes Kings Law Coefficients from a CSV and record the data from a hotwire and output the raw voltages as well as velocities. The turbulent kinetic energy and power spectral density will be added to this document soon.
