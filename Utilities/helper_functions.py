# Developed By Code|<Ill at 6/30/2019
# Developed VM IP 203.241.246.158

import pandas as pd
import numpy as np
import math
import scipy.stats
from scipy.signal import butter, lfilter, freqz
from sklearn.metrics import accuracy_score, f1_score

#Resultant of Three Axws
def resultant(axis_x, axis_y, axis_z):
    '''
    :param axis_x: Axis of the Sensor
    :param axis_y: Axis of the Sensor
    :param axis_z: Axis of the Sensor
    :return: Single Array
    '''
    axis_x=np.array(axis_x)
    axis_y=np.array(axis_y)
    axis_z=np.array(axis_z)
    resultant_vec=[]
    adder=((axis_x*axis_x)+(axis_y*axis_y)+(axis_z*axis_z))
    for i in adder:
        resultant_vec.append(math.sqrt(i))

    return resultant_vec


#Filtering of The Signals
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def Label_to_Num(var_arr):
    conv_array=[]

    for i in var_arr:
        if(i=='Walk'):
            conv_array.append(int(0))
        elif(i=='Sit'):
            conv_array.append(int(1))
        elif (i == 'Stay'):
            conv_array.append(int(2))
        elif (i == 'Eat'):
            conv_array.append(int(3))
        elif (i == 'Sideway'):
            conv_array.append(int(4))
        elif (i == 'Jump'):
            conv_array.append(int(5))
        elif (i == 'Nosework'):
            conv_array.append(int(6))
        else:
            conv_array.append(int(100))# Exception Handling

    return conv_array

def Num_to_Label(var_arr):
    conv_array=[]

    for i in var_arr:
        if (i==0):
            conv_array.append('Walk')
        elif (i==1):
            conv_array.append('Sit')
        elif (i == 2):
            conv_array.append('Stay')
        elif (i == 3):
            conv_array.append('Eat')
        elif (i == 4):
            conv_array.append('Sideway')
        elif (i == 5):
            conv_array.append('Jump')
        elif (i == 6):
            conv_array.append('Nosework')
        else:
            conv_array.append('Wrong_Label')

    return conv_array


def leastFrequent(arr, n=165):
    # Sort the array
    arr.sort()

    # find the min frequency using
    # linear traversal
    min_count = n + 1
    res = -1
    curr_count = 1
    for i in range(1, n):
        if (arr[i] == arr[i - 1]):
            curr_count = curr_count + 1
        else:
            if (curr_count < min_count):
                min_count = curr_count
                res = arr[i - 1]

            curr_count = 1

    # If last element is least frequent
    if (curr_count < min_count):
        min_count = curr_count
        res = arr[n - 1]

    return res

def calc_accuracy_per_sample(data):
    exhaust_list=[]

    activity_list=data['Information'].unique()

    for i in activity_list:

        activity_data=data[data['Information']==i]

        Sample_Arr=[]
        Accuracy_Arr=[]
        F1_micro = []

        for j in range(1,166):
            actual=activity_data[activity_data['Sample']==j]['Label']
            prediction=activity_data[activity_data['Sample']==j]['Predictions']

            accuracy=accuracy_score(actual,prediction)
            f_micro = f1_score(actual, prediction, average='micro')

            accuracy=round(accuracy*100,2)
            f_micro = round(f_micro * 100, 2)

            Sample_Arr.append(j)
            Accuracy_Arr.append(accuracy)
            F1_micro.append(f_micro)


        acc_perf=dict(zip(Sample_Arr,Accuracy_Arr))
        f_perf = dict(zip(Sample_Arr, F1_micro))

        activity_data['Accuracy']=activity_data['Sample'].map(acc_perf)
        activity_data['F1-Score'] = activity_data['Sample'].map(f_perf)

        grouped_data=activity_data.groupby('Sample',as_index=False).agg({"Information":lambda x: scipy.stats.mode(x)[0],"Accuracy":lambda x: scipy.stats.mode(x)[0],"F1-Score":lambda x: scipy.stats.mode(x)[0]})

        exhaust_list.append(grouped_data)

    transfer_data=pd.concat(exhaust_list, axis=0, ignore_index=True)

    return transfer_data

def calc_accuracy_per_activity(data):
    Activity_Arr = []
    Accuracy_Arr = []
    F1_micro=[]

    activity_list = data['Information'].unique()

    for i in activity_list:
        actual = data[data['Information'] == i]['Label']
        prediction = data[data['Information'] == i]['Predictions']

        accuracy=accuracy_score(actual,prediction)
        f_micro=f1_score(actual, prediction, average='micro')

        accuracy = round(accuracy * 100, 2)
        f_micro = round(f_micro * 100, 2)

        Activity_Arr.append(i)
        Accuracy_Arr.append(accuracy)
        F1_micro.append(f_micro)

    acc_perf=dict(zip(Activity_Arr, Accuracy_Arr))
    f_perf=dict(zip(Activity_Arr, F1_micro))

    data['Accuracy']=data['Information'].map(acc_perf)
    data['F1-Score']=data['Information'].map(f_perf)

    grouped_data = data.groupby('Information', as_index=False).agg({"Accuracy": lambda x: scipy.stats.mode(x)[0], "F1-Score":lambda x: scipy.stats.mode(x)[0]})

    return grouped_data