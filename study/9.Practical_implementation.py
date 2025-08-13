# In[ ]:
import sys
import time
import  matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from scipy import signal
import numpy as np
from scipy.integrate import simps
from scipy import signal
import re

# In[ ]:
from google.colab import drive
drive.mount('/content/drive')

# In[ ]:
data_1_morning = "/content/drive/MyDrive/EEG_course/dataset/dataset_emotional/1_morning.xlsx"
data_1_evening = "/content/drive/MyDrive/EEG_course/dataset/dataset_emotional/1_evening.xlsx"
data_2_morning = "/content/drive/MyDrive/EEG_course/dataset/dataset_emotional/2_morning.xlsx"
data_2_evening = "/content/drive/MyDrive/EEG_course/dataset/dataset_emotional/2_evening.xlsx"
data_3_morning = "/content/drive/MyDrive/EEG_course/dataset/dataset_emotional/3_morning.xlsx"
data_3_evening = "/content/drive/MyDrive/EEG_course/dataset/dataset_emotional/3_evening.xlsx"
data_4_morning = "/content/drive/MyDrive/EEG_course/dataset/dataset_emotional/4_morning.xlsx"
data_4_evening = "/content/drive/MyDrive/EEG_course/dataset/dataset_emotional/4_evening.xlsx"
data_5_morning = "/content/drive/MyDrive/EEG_course/dataset/dataset_emotional/5_morning.xlsx"
data_5_evening = "/content/drive/MyDrive/EEG_course/dataset/dataset_emotional/5_evening.xlsx"


data = pd.read_excel(data_1_morning)
print(data.columns)
print(data.shape)
data = data["ch1"]
plt.xlabel('Samples')
plt.ylabel('EEG, μV')
plt.title('EEG sample')
plt.plot(data[2000:8000]) # use the clean data
plt.show()

# In[ ]:
"""
The script iterates over increasing data segments, calculates their differential entropy,
 and stores these values. It then fits a polynomial regression to the entropy data
 and plots both the entropy values
and the fitted polynomial curve to visualize the relationship between data EEG in different time and entropy.
"""
from scipy.stats import differential_entropy, norm

data_ent_graph = []

for a in range (1000, len(data), 1000):
    values = data[:a]
    data_ent_final = differential_entropy(values)
    data_ent_graph.append(data_ent_final)

x_inter_removed_high = []
for data_x in range (0, len(data_ent_graph),1):
    x_inter_removed_high.append(data_x)

degree = 5

# Polynomial regression
coefficients = np.polyfit(x_inter_removed_high, data_ent_graph, degree)
# Polynomial coefficients
y_poly = np.polyval(coefficients, x_inter_removed_high)

plt.xlabel('Data step')
plt.ylabel('Entropy')
plt.plot(x_inter_removed_high, data_ent_graph, 'o', label='Data Points')
plt.plot(x_inter_removed_high, y_poly, 'r', label=f'Polynomial Regression (Degree {degree})')
plt.legend()
plt.show()

# In[ ]:
"""
This script calculates and visualizes the power spectral density (PSD) of a data segment using Welch's method,
highlights the alpha frequency band, and computes the absolute power within this band.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.integrate import simps

data_short = data[1000:8000]

#sns.set(font_scale=1.2)

# Define sampling frequency and time vector
# convert samples to time

# Define window length (4 seconds)
sf = 250
win = 4 * sf
freqs, psd = signal.welch(data_short, sf, nperseg=win)

# Plot the power spectrum
sns.set(font_scale=1.2, style='white')

# Define delta lower and upper limits
low, high = 8, 13  # Delta Waves: Up to 4 Hz
                   # Alpha Waves: 8 - 13 Hz
                   # Theta Waves: 4-7 Hz
                   # Gamma Waves: 30-100 Hz

# Find intersecting values in frequency vector
idx_delta = np.logical_and(freqs >= low, freqs <= high)

# Plot the power spectral density and fill the delta area
plt.figure(figsize=(7, 4))
plt.plot(freqs, psd, lw=2, color='k')

plt.fill_between(freqs, psd, where=idx_delta, color='green')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (uV^2 / Hz)')
plt.xlim([0, 20])
plt.ylim([0, psd.max() * 1.1])
plt.title("Welch's periodogram")
plt.show()


# Frequency resolution
freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25

# Compute the absolute power by approximating the area under the curve
delta_power = simps(psd[idx_delta], dx=freq_res)

print('Absolute delta power: %.3f uV^2' % delta_power)

# In[ ]:


