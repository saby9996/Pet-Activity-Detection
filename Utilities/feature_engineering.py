# Developed By Code|<Ill at 7/8/2019
# Developed VM IP 203.241.246.158

import numpy as np
import pandas as pd
from scipy import stats


#Helper Function
## Feature Engineering

def mean(x, y, z):
    """Calculates mean"""
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    mean_z = np.mean(z)
    return mean_x, mean_y, mean_z


def std_dev(x, y, z):
    """Calculates standard deviation"""
    std_x = np.std(x)
    std_y = np.std(y)
    std_z = np.std(z)
    return std_x, std_y, std_z


def mad(x, y, z):
    """Calculates median absolute deviation"""
    mad_x = np.median(np.abs(x - np.median(x)))
    mad_y = np.median(np.abs(y - np.median(y)))
    mad_z = np.median(np.abs(z - np.median(z)))
    return mad_x, mad_y, mad_z


def minimum(x, y, z):
    """Calculates minimum"""
    return min(x), min(y), min(z)


def maximum(x, y, z):
    """Calculates maximum"""
    return max(x), max(y), max(z)


def energy_measure(x, y, z):
    """Calculates energy measures"""
    em_x = np.mean(np.square(x))
    em_y = np.mean(np.square(y))
    em_z = np.mean(np.square(z))
    return em_x, em_y, em_z


def inter_quartile_range(x, y, z):
    """Calculates inter-quartile range"""
    iqr_x = np.subtract(*np.percentile(x, [75, 25]))
    iqr_y = np.subtract(*np.percentile(y, [75, 25]))
    iqr_z = np.subtract(*np.percentile(z, [75, 25]))
    return iqr_x, iqr_y, iqr_z


def sma(x, y, z):
    """Calculates signal magnitude area"""
    abs_x = np.absolute(x)
    abs_y = np.absolute(y)
    abs_z = np.absolute(z)
    return np.mean(abs_x + abs_y + abs_z)


def skewness(x, y, z):
    """Calculates skewness"""
    skew_x = stats.skew(x)
    skew_y = stats.skew(y)
    skew_z = stats.skew(z)
    return skew_x, skew_y, skew_z


def kurt(x, y, z):
    """Calculates kurtosis"""
    kurt_x = stats.kurtosis(x, fisher=False)
    kurt_y = stats.kurtosis(y, fisher=False)
    kurt_z = stats.kurtosis(z, fisher=False)
    return kurt_x, kurt_y, kurt_z


#Rolling Methods
# Rolling Mean
def rolling_mean(x, y, z, window=3):
    zeros = np.repeat(0, window - 1, axis=0)  # Zeroes is The Number of zeroes To Be Added

    x_list = np.concatenate([zeros, x])  # Concatenated List With Zeroes in the Begining X
    y_list = np.concatenate([zeros, y])  # Concatenated List With Zeroes in the Begining Y
    z_list = np.concatenate([zeros, z])  # Concatenated List With Zeroes in the Begining Z

    x_conv = []
    y_conv = []
    z_conv = []

    for i in range(len(x)):
        x_net = x_list[i:i + window]
        y_net = y_list[i:i + window]
        z_net = z_list[i:i + window]

        x_val, y_val, z_val = mean(x_net, y_net, z_net)

        x_conv.append(x_val)
        y_conv.append(y_val)
        z_conv.append(z_val)

    return x_conv, y_conv, z_conv


# Rolling Standard Deviation
def rolling_std(x, y, z, window=3):
    zeros = np.repeat(0, window - 1, axis=0)  # Zeroes is The Number of zeroes To Be Added

    x_list = np.concatenate([zeros, x])  # Concatenated List With Zeroes in the Begining X
    y_list = np.concatenate([zeros, y])  # Concatenated List With Zeroes in the Begining Y
    z_list = np.concatenate([zeros, z])  # Concatenated List With Zeroes in the Begining Z

    x_conv = []
    y_conv = []
    z_conv = []

    for i in range(len(x)):
        x_net = x_list[i:i + window]
        y_net = y_list[i:i + window]
        z_net = z_list[i:i + window]

        x_val, y_val, z_val = std_dev(x_net, y_net, z_net)

        x_conv.append(x_val)
        y_conv.append(y_val)
        z_conv.append(z_val)

    return x_conv, y_conv, z_conv


# Rolling Mean Absolute Deviation
def rolling_mad(x, y, z, window=3):
    zeros = np.repeat(0, window - 1, axis=0)  # Zeroes is The Number of zeroes To Be Added

    x_list = np.concatenate([zeros, x])  # Concatenated List With Zeroes in the Begining X
    y_list = np.concatenate([zeros, y])  # Concatenated List With Zeroes in the Begining Y
    z_list = np.concatenate([zeros, z])  # Concatenated List With Zeroes in the Begining Z

    x_conv = []
    y_conv = []
    z_conv = []

    for i in range(len(x)):
        x_net = x_list[i:i + window]
        y_net = y_list[i:i + window]
        z_net = z_list[i:i + window]

        x_val, y_val, z_val = mad(x_net, y_net, z_net)

        x_conv.append(x_val)
        y_conv.append(y_val)
        z_conv.append(z_val)

    return x_conv, y_conv, z_conv


# Rolling Minimum
def rolling_min(x, y, z, window=3):
    zeros = np.repeat(0, window - 1, axis=0)  # Zeroes is The Number of zeroes To Be Added

    x_list = np.concatenate([zeros, x])  # Concatenated List With Zeroes in the Begining X
    y_list = np.concatenate([zeros, y])  # Concatenated List With Zeroes in the Begining Y
    z_list = np.concatenate([zeros, z])  # Concatenated List With Zeroes in the Begining Z

    x_conv = []
    y_conv = []
    z_conv = []

    for i in range(len(x)):
        x_net = x_list[i:i + window]
        y_net = y_list[i:i + window]
        z_net = z_list[i:i + window]

        x_val, y_val, z_val = minimum(x_net, y_net, z_net)

        x_conv.append(x_val)
        y_conv.append(y_val)
        z_conv.append(z_val)

    return x_conv, y_conv, z_conv


# Rolling Maximum
def rolling_max(x, y, z, window=3):
    zeros = np.repeat(0, window - 1, axis=0)  # Zeroes is The Number of zeroes To Be Added

    x_list = np.concatenate([zeros, x])  # Concatenated List With Zeroes in the Begining X
    y_list = np.concatenate([zeros, y])  # Concatenated List With Zeroes in the Begining Y
    z_list = np.concatenate([zeros, z])  # Concatenated List With Zeroes in the Begining Z

    x_conv = []
    y_conv = []
    z_conv = []

    for i in range(len(x)):
        x_net = x_list[i:i + window]
        y_net = y_list[i:i + window]
        z_net = z_list[i:i + window]

        x_val, y_val, z_val = maximum(x_net, y_net, z_net)

        x_conv.append(x_val)
        y_conv.append(y_val)
        z_conv.append(z_val)

    return x_conv, y_conv, z_conv


# Rolling Energy MEasure
def rolling_energy_measure(x, y, z, window=3):
    zeros = np.repeat(0, window - 1, axis=0)  # Zeroes is The Number of zeroes To Be Added

    x_list = np.concatenate([zeros, x])  # Concatenated List With Zeroes in the Begining X
    y_list = np.concatenate([zeros, y])  # Concatenated List With Zeroes in the Begining Y
    z_list = np.concatenate([zeros, z])  # Concatenated List With Zeroes in the Begining Z

    x_conv = []
    y_conv = []
    z_conv = []

    for i in range(len(x)):
        x_net = x_list[i:i + window]
        y_net = y_list[i:i + window]
        z_net = z_list[i:i + window]

        x_val, y_val, z_val = energy_measure(x_net, y_net, z_net)

        x_conv.append(x_val)
        y_conv.append(y_val)
        z_conv.append(z_val)

    return x_conv, y_conv, z_conv


# Rolling Inter Quartile Range
def rolling_IQ_Range(x, y, z, window=3):
    zeros = np.repeat(0, window - 1, axis=0)  # Zeroes is The Number of zeroes To Be Added

    x_list = np.concatenate([zeros, x])  # Concatenated List With Zeroes in the Begining X
    y_list = np.concatenate([zeros, y])  # Concatenated List With Zeroes in the Begining Y
    z_list = np.concatenate([zeros, z])  # Concatenated List With Zeroes in the Begining Z

    x_conv = []
    y_conv = []
    z_conv = []

    for i in range(len(x)):
        x_net = x_list[i:i + window]
        y_net = y_list[i:i + window]
        z_net = z_list[i:i + window]

        x_val, y_val, z_val = inter_quartile_range(x_net, y_net, z_net)

        x_conv.append(x_val)
        y_conv.append(y_val)
        z_conv.append(z_val)

    return x_conv, y_conv, z_conv


# Rolling Skewness
def rolling_skew(x, y, z, window=3):
    zeros = np.repeat(0, window - 1, axis=0)  # Zeroes is The Number of zeroes To Be Added

    x_list = np.concatenate([zeros, x])  # Concatenated List With Zeroes in the Begining X
    y_list = np.concatenate([zeros, y])  # Concatenated List With Zeroes in the Begining Y
    z_list = np.concatenate([zeros, z])  # Concatenated List With Zeroes in the Begining Z

    x_conv = []
    y_conv = []
    z_conv = []

    for i in range(len(x)):
        x_net = x_list[i:i + window]
        y_net = y_list[i:i + window]
        z_net = z_list[i:i + window]

        x_val, y_val, z_val = skewness(x_net, y_net, z_net)

        x_conv.append(x_val)
        y_conv.append(y_val)
        z_conv.append(z_val)

    return x_conv, y_conv, z_conv


# Rolling Kurtosis
def rolling_kurt(x, y, z, window=3):
    zeros = np.repeat(0, window - 1, axis=0)  # Zeroes is The Number of zeroes To Be Added

    x_list = np.concatenate([zeros, x])  # Concatenated List With Zeroes in the Begining X
    y_list = np.concatenate([zeros, y])  # Concatenated List With Zeroes in the Begining Y
    z_list = np.concatenate([zeros, z])  # Concatenated List With Zeroes in the Begining Z

    x_conv = []
    y_conv = []
    z_conv = []

    for i in range(len(x)):
        x_net = x_list[i:i + window]
        y_net = y_list[i:i + window]
        z_net = z_list[i:i + window]

        x_val, y_val, z_val = kurt(x_net, y_net, z_net)

        x_conv.append(x_val)
        y_conv.append(y_val)
        z_conv.append(z_val)

    return x_conv, y_conv, z_conv



# Routine Feature Engineering
def feature_data(data):
    data['roll_mean_NAccX'], data['roll_mean_NAccY'], data['roll_mean_NAccZ'] = rolling_mean(data['N-AccX'],data['N-AccY'],data['N-AccZ'])
    data['roll_std_NAccX'], data['roll_std_NAccY'], data['roll_std_NAccZ'] = rolling_std(data['N-AccX'], data['N-AccY'],data['N-AccZ'])
    data['roll_mad_NAccX'], data['roll_mad_NAccY'], data['roll_mad_NAccZ'] = rolling_mad(data['N-AccX'], data['N-AccY'],data['N-AccZ'])
    data['roll_min_NAccX'], data['roll_min_NAccY'], data['roll_min_NAccZ'] = rolling_min(data['N-AccX'], data['N-AccY'],data['N-AccZ'])
    data['roll_max_NAccX'], data['roll_max_NAccY'], data['roll_max_NAccZ'] = rolling_max(data['N-AccX'], data['N-AccY'],data['N-AccZ'])
    data['roll_EME_NAccX'], data['roll_EME_NAccY'], data['roll_EME_NAccZ'] = rolling_energy_measure(data['N-AccX'],data['N-AccY'],data['N-AccZ'])
    data['roll_IQR_NAccX'], data['roll_IQR_NAccY'], data['roll_IQR_NAccZ'] = rolling_IQ_Range(data['N-AccX'],data['N-AccY'],data['N-AccZ'])
    data['N-Acc-SMA'] = sma(data['N-AccX'], data['N-AccY'], data['N-AccZ'])
    data['roll_skew_NAccX'], data['roll_skew_NAccY'], data['roll_skew_NAccZ'] = rolling_skew(data['N-AccX'],data['N-AccY'],data['N-AccZ'])
    data['roll_kurt_NAccX'], data['roll_kurt_NAccY'], data['roll_kurt_NAccZ'] = rolling_kurt(data['N-AccX'],data['N-AccY'],data['N-AccZ'])

    data['roll_mean_NGyroX'], data['roll_mean_NGyroY'], data['roll_mean_NGyroZ'] = rolling_mean(data['N-GyroX'],data['N-GyroY'],data['N-GyroZ'])
    data['roll_std_NGyroX'], data['roll_std_NGyroY'], data['roll_std_NGyroZ'] = rolling_std(data['N-GyroX'],data['N-GyroY'],data['N-GyroZ'])
    data['roll_mad_NGyroX'], data['roll_mad_NGyroY'], data['roll_mad_NGyroZ'] = rolling_mad(data['N-GyroX'],data['N-GyroY'],data['N-GyroZ'])
    data['roll_min_NGyroX'], data['roll_min_NGyroY'], data['roll_min_NGyroZ'] = rolling_min(data['N-GyroX'],data['N-GyroY'],data['N-GyroZ'])
    data['roll_max_NGyroX'], data['roll_max_NGyroY'], data['roll_max_NGyroZ'] = rolling_max(data['N-GyroX'],data['N-GyroY'],data['N-GyroZ'])
    data['roll_EME_NGyroX'], data['roll_EME_NGyroY'], data['roll_EME_NGyroZ'] = rolling_energy_measure(data['N-GyroX'],data['N-GyroY'],data['N-GyroZ'])
    data['roll_IQR_NGyroX'], data['roll_IQR_NGyroY'], data['roll_IQR_NGyroZ'] = rolling_IQ_Range(data['N-GyroX'],data['N-GyroY'],data['N-GyroZ'])
    data['N-Gyro-SMA'] = sma(data['N-GyroX'], data['N-GyroY'], data['N-GyroZ'])
    data['roll_skew_NGyroX'], data['roll_skew_NGyroY'], data['roll_skew_NGyroZ'] = rolling_skew(data['N-GyroX'],data['N-GyroY'],data['N-GyroZ'])
    data['roll_kurt_NGyroX'], data['roll_kurt_NGyroY'], data['roll_kurt_NGyroZ'] = rolling_kurt(data['N-GyroX'],data['N-GyroY'],data['N-GyroZ'])

    data['roll_mean_TAccX'], data['roll_mean_TAccY'], data['roll_mean_TAccZ'] = rolling_mean(data['T-AccX'],data['T-AccY'],data['T-AccZ'])
    data['roll_std_TAccX'], data['roll_std_TAccY'], data['roll_std_TAccZ'] = rolling_std(data['T-AccX'], data['T-AccY'],data['T-AccZ'])
    data['roll_mad_TAccX'], data['roll_mad_TAccY'], data['roll_mad_TAccZ'] = rolling_mad(data['T-AccX'], data['T-AccY'],data['T-AccZ'])
    data['roll_min_TAccX'], data['roll_min_TAccY'], data['roll_min_TAccZ'] = rolling_min(data['T-AccX'], data['T-AccY'],data['T-AccZ'])
    data['roll_max_TAccX'], data['roll_max_TAccY'], data['roll_max_TAccZ'] = rolling_max(data['T-AccX'], data['T-AccY'],data['T-AccZ'])
    data['roll_EME_TAccX'], data['roll_EME_TAccY'], data['roll_EME_TAccZ'] = rolling_energy_measure(data['T-AccX'],data['T-AccY'],data['T-AccZ'])
    data['roll_IQR_TAccX'], data['roll_IQR_TAccY'], data['roll_IQR_TAccZ'] = rolling_IQ_Range(data['T-AccX'],data['T-AccY'],data['T-AccZ'])
    data['T-Acc-SMA'] = sma(data['T-AccX'], data['T-AccY'], data['T-AccZ'])
    data['roll_skew_TAccX'], data['roll_skew_TAccY'], data['roll_skew_TAccZ'] = rolling_skew(data['T-AccX'],data['T-AccY'],data['T-AccZ'])
    data['roll_kurt_TAccX'], data['roll_kurt_TAccY'], data['roll_kurt_TAccZ'] = rolling_kurt(data['T-AccX'],data['T-AccY'],data['T-AccZ'])

    data['roll_mean_TGyroX'], data['roll_mean_TGyroY'], data['roll_mean_TGyroZ'] = rolling_mean(data['T-GyroX'],data['T-GyroY'],data['T-GyroZ'])
    data['roll_std_TGyroX'], data['roll_std_TGyroY'], data['roll_std_TGyroZ'] = rolling_std(data['T-GyroX'],data['T-GyroY'],data['T-GyroZ'])
    data['roll_mad_TGyroX'], data['roll_mad_TGyroY'], data['roll_mad_TGyroZ'] = rolling_mad(data['T-GyroX'],data['T-GyroY'],data['T-GyroZ'])
    data['roll_min_TGyroX'], data['roll_min_TGyroY'], data['roll_min_TGyroZ'] = rolling_min(data['T-GyroX'],data['T-GyroY'],data['T-GyroZ'])
    data['roll_max_TGyroX'], data['roll_max_TGyroY'], data['roll_max_TGyroZ'] = rolling_max(data['T-GyroX'],data['T-GyroY'],data['T-GyroZ'])
    data['roll_EME_TGyroX'], data['roll_EME_TGyroY'], data['roll_EME_TGyroZ'] = rolling_energy_measure(data['T-GyroX'],data['T-GyroY'],data['T-GyroZ'])
    data['roll_IQR_TGyroX'], data['roll_IQR_TGyroY'], data['roll_IQR_TGyroZ'] = rolling_IQ_Range(data['T-GyroX'],data['T-GyroY'],data['T-GyroZ'])
    data['T-Gyro-SMA'] = sma(data['T-GyroX'], data['T-GyroY'], data['T-GyroZ'])
    data['roll_skew_TGyroX'], data['roll_skew_TGyroY'], data['roll_skew_TGyroZ'] = rolling_skew(data['T-GyroX'],data['T-GyroY'],data['T-GyroZ'])
    data['roll_kurt_TGyroX'], data['roll_kurt_TGyroY'], data['roll_kurt_TGyroZ'] = rolling_kurt(data['T-GyroX'],data['T-GyroY'],data['T-GyroZ'])

    return data
