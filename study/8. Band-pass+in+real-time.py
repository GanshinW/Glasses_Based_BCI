import pandas as pd
from scipy import signal
import numpy as np
import time
import pylab as pl
from IPython import display

from google.colab import drive
drive.mount('/content/drive')
"""
This script processes real-time EEG data using a Butterworth bandpass filter.
The butter_bandpass function designs the filter by calculating the normalized cutoff frequencies and using the scipy.signal.butter function.
The butter_bandpass_filter function applies this filter to the data.
The main loop reads EEG data from an Excel file in chunks, initially gathering enough data for the filter, then continually adding new data.
Each new dataset is combined with the previous filtered data, processed through the bandpass filter, and plotted in real-time to visualize the filtered and raw EEG signals.
This process repeats indefinitely, simulating a real-time data processing pipeline.
"""


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    data = signal.lfilter(b, a, data)
    return data

dataset_y = pd.read_excel("/content/drive/MyDrive/EEG_course/dataset/dataset.xlsx")

channel = 0

dataset_y_row = dataset_y.iloc[channel]
print (dataset_y_row.shape)
dataset_y = dataset_y.transpose()
dataset_y = dataset_y.iloc[0]
just_one_time = 0
data_before = []
data_after = []
fps = 250
cutoff = 1
cutoffs = 10


data_lenght_for_Filter = 4     # how much we read data for filter, all lenght  [_____] + [_____] + [_____]
read_data_lenght_one_time = 1   # for one time how much read  [_____]

start = 0
total_lenght = end = 500


while 1:
        if just_one_time == 0:
            for b in range (0,data_lenght_for_Filter,1):
                for a in range (0,read_data_lenght_one_time,1):
                    data_read = dataset_y[start:end]
                    start = start + 250
                    end = end + 250
                    data_before.append(data_read)

            just_one_time = 1
            data_before = data_before[read_data_lenght_one_time:]

        for c in range (0,read_data_lenght_one_time,1):
            data_read = dataset_y[start:end]
            start = start + 250
            end = end + 250
            data_after.append(data_read)

        data_before_for_sum = data_before
        data_after_for_sum = data_after

        data_before_for_sum = [item for sublist in data_before for item in sublist]
        data_after_for_sum = [item for sublist in data_after for item in sublist]
        dataset =  data_before_for_sum + data_after_for_sum #+ data_after_flip
        dataset = [int(x) for x in dataset]

        dataset_before = data_before + data_after
        data_before = dataset_before[read_data_lenght_one_time:]
        data = butter_bandpass_filter(dataset, cutoff, cutoffs,fps)

        data_after = []
        dataset = []


        pl.clf()
        pl.plot(data[-total_lenght:],label='Data after band-pass filter')
        data_for_graph = np.array(data_read)
        pl.title("Real-time for band-pass filter , Channel " + str(channel + 1))
        pl.ylabel('EEG, µV')
        pl.xlabel('Sample')
        pl.plot(data_for_graph - np.average(data_for_graph), label='Raw data')
        pl.legend()

        display.display(pl.gcf())
        display.clear_output(wait=True)


        time.sleep(1)