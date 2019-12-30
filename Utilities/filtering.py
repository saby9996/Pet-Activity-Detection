# Developed By Code|<Ill at 7/8/2019
# Developed VM IP 203.241.246.158
import pandas as pd
import numpy as np
#Self Declared Modules and Packages
from Utilities import helper_functions


def filter_data(data):
    # Function Call For Filtering Accelerometer Signals
    data['N-AccX'] = helper_functions.butter_lowpass_filter(data['N-AccX'], cutoff=3.667, fs=33.3, order=5)
    data['N-AccY'] = helper_functions.butter_lowpass_filter(data['N-AccY'], cutoff=3.667, fs=33.3, order=5)
    data['N-AccZ'] = helper_functions.butter_lowpass_filter(data['N-AccZ'], cutoff=3.667, fs=33.3, order=5)

    data['N-GyroX'] = helper_functions.butter_lowpass_filter(data['N-GyroX'], cutoff=3.667, fs=33.3, order=5)
    data['N-GyroY'] = helper_functions.butter_lowpass_filter(data['N-GyroY'], cutoff=3.667, fs=33.3, order=5)
    data['N-GyroZ'] = helper_functions.butter_lowpass_filter(data['N-GyroZ'], cutoff=3.667, fs=33.3, order=5)

    data['T-AccX'] = helper_functions.butter_lowpass_filter(data['T-AccX'], cutoff=3.667, fs=33.3, order=5)
    data['T-AccY'] = helper_functions.butter_lowpass_filter(data['T-AccY'], cutoff=3.667, fs=33.3, order=5)
    data['T-AccZ'] = helper_functions.butter_lowpass_filter(data['T-AccZ'], cutoff=3.667, fs=33.3, order=5)

    data['T-GyroX'] = helper_functions.butter_lowpass_filter(data['T-GyroX'], cutoff=3.667, fs=33.3, order=5)
    data['T-GyroY'] = helper_functions.butter_lowpass_filter(data['T-GyroY'], cutoff=3.667, fs=33.3, order=5)
    data['T-GyroZ'] = helper_functions.butter_lowpass_filter(data['T-GyroZ'], cutoff=3.667, fs=33.3, order=5)

    return data