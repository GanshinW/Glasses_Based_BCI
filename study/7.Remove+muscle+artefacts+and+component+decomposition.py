# %% [markdown]
# # Chapter 4. Muscle artefacts removal (component decomposition)

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ## Import libraries
# To install libraries in Python, you can use a package manager like pip, which comes pre-installed with most Python distributions.
# 

# %%
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.interpolate as spi
from scipy import signal
from sklearn.decomposition import PCA,FastICA


! pip install pyts
from pyts.decomposition import SingularSpectrumAnalysis

!pip install EMD-signal
from PyEMD import EEMD
# from pyhht.emd import EMD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# %% [markdown]
# ## Import Dataset

# %% [markdown]
# Load dataset https://github.com/Ildaron/ironbci

# %%
# load the Excel file as a DataFrame
eeg_data = pd.read_excel("/content/drive/MyDrive/EEG_course/dataset/dataset.xlsx")
display(eeg_data.head())

# %%
# select the data from the first channel
channel = 0
channel_data = eeg_data.iloc[:, channel]
# convert from Digital Value of Analog Digital converter (ADC) ADS1299 to microvolts µV
channel_data = round(1000000 * 4.5 * (channel_data / 16777215), 2)

# plot the EEG data
plt.plot(channel_data)
plt.title(f"EEG, Raw data, Channel {channel+1}")
plt.ylabel('EEG, µV')  # Data from ADS1299
plt.xlabel('Sample')
plt.show()

# %% [markdown]
# ## Band Pass Filter
# 
# Band-pass filtering is a common signal processing technique used in EEG (Electroencephalography) data analysis to isolate specific frequency bands of interest while removing unwanted frequencies. Band-pass filtering allows researchers to focus on particular brainwave rhythms that are relevant to their study. For example, you might want to extract the alpha, beta, or gamma waves from the EEG data.  
# 
# You can read more about band-pass filter in [Chapter 1](https://graceful-kelpie-579688.netlify.app/chapters/chapter_1-band_pass_filter).

# %%
def butter_highpass_filter(data, cutoff, nyq, order=5):
    """Butterworth high-pass filter.
    Args:
        data (array_like): data to be filtered.
        cutoff (float): cutoff frequency.
        order (int): order of the filter.
    Returns:
        array: filtered data."""
    normal_cutoff = cutoff / nyq  # normalized cutoff frequency
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def butter_lowpass_filter(data, cutoff, nyq, order=5):
    """Butterworth low-pass filter.
    Args:
        data (array_like): data to be filtered.
        cutoff (float): cutoff frequency.
        order (int): order of the filter.
    Returns:
        array: filtered data."""
    normal_cutoff = cutoff / nyq  # normalized cutoff frequency
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data

channel_data_bp_filtered = channel_data.copy()  # copy the data
fps = 250  # sampling frequency
cutoff_high = 1  # cutoff frequency of the high-pass filter
cutoff_low = 15  # cutoff frequency of the low-pass filter
nyq = 0.5 * fps  # Nyquist frequency (half of the sampling frequency)

# apply the band-pass filter
channel_data_bp_filtered = butter_highpass_filter(
    data=channel_data_bp_filtered,
    cutoff=cutoff_high,
    nyq=nyq,
    order=5)
channel_data_bp_filtered = butter_lowpass_filter(
    data=channel_data_bp_filtered,
    cutoff=cutoff_low,
    nyq=nyq,
    order=4)

plt.plot(channel_data_bp_filtered)
plt.title("Data after Band-pass Filter, Channel " +  str(channel+1) +", ("+str(cutoff_high)+"-"+str(cutoff_low) + "Hz)")
plt.ylabel('EEG, µV')
plt.xlabel('Sample')
plt.show()






# %%
# apply the band-pass filter for the whole dataset
data_bp_filtered = eeg_data.copy()  # copy the data

for ch in range(data_bp_filtered.shape[1]):
    data_bp_filtered.iloc[:, ch] = butter_highpass_filter(
        data=data_bp_filtered.iloc[:, ch],
        cutoff=cutoff_high,
        nyq=nyq,
        order=5)
    data_bp_filtered.iloc[:, ch] = butter_lowpass_filter(
        data=data_bp_filtered.iloc[:, ch],
        cutoff=cutoff_low,
        nyq=nyq,
        order=4)

# %% [markdown]
# ## Remove muscle artifacts (component decomposition)
# 
# Removing muscle artifacts from EEG (Electroencephalography) data is a crucial preprocessing step to obtain clean and reliable brain signals. Muscle artifacts in EEG are caused by electrical activity from muscle contractions and can contaminate the brainwave recordings, making it challenging to analyze brain activity accurately.
# 
# Component decomposition techniques are advantageous for EEG analysis because they can effectively separate mixed signals into their underlying sources. This helps in isolating genuine neural activity from various artifacts, leading to cleaner and more interpretable EEG data. Such methods enhance the accuracy of brain signal analysis, improve the sensitivity to subtle neural patterns, and contribute to the reliability of EEG findings, making them essential tools in neuroscience research.

# %% [markdown]
# ## Independent Component Analysis (ICA)
# 
# Independent Component Analysis (ICA) is a widely used signal processing technique for separating mixed sources in EEG data. It is particularly useful for decomposing EEG signals into their underlying independent components, which can represent different brain sources or artifacts. ICA helps to remove artifacts and identify specific brain-related activity, making it an essential tool for EEG data analysis.
# 
# **Positive**
# 
# *   Statistical Independence: ICA is designed to decompose mixed signals into components that are statistically independent. Muscle artifacts, which often have distinct temporal patterns and different frequency characteristics from neural signals, can be modeled as separate independent components.
# 
# * Multichannel Data: EEG signals are recorded from multiple channels, capturing spatially distributed neural and artifact sources. ICA's ability to simultaneously analyze multichannel data helps in separating out spatially localized muscle artifacts.
# 
# * Unmixing Overlapping Sources: EEG signals are a mixture of various sources, including neural activity, noise, and artifacts. ICA can unmix these overlapping sources by identifying their unique statistical properties, effectively isolating the muscle artifacts.
# 
# * Adaptive Filtering: ICA adaptively identifies artifact sources without relying on predefined templates or assumptions about their characteristics. This makes it versatile and effective in capturing a wide range of muscle artifact patterns.
# 
# * Reduced Dependence on Reference Electrodes: Unlike methods that rely on reference electrodes, ICA does not require specific electrode placements or reference strategies. This makes it useful for removing muscle artifacts even in scenarios where reference electrode placement might be suboptimal.
# 
# * Non-stationary Artifacts: Muscle artifacts can exhibit non-stationary characteristics, such as changes in amplitude or frequency. ICA's flexibility in capturing time-varying source patterns makes it suitable for removing such non-stationary artifacts.
# 
# * Enhanced Data Quality: By separating muscle artifacts from neural signals, ICA improves the overall quality of the EEG data, making subsequent analyses more accurate and reliable.
# 
# **Negative**
# 
# *   Assumption Violations: ICA assumes statistical independence of sources, which might not hold true for all types of artifacts. Muscle artifacts, in particular, can sometimes exhibit temporal or spectral correlations that violate this assumption.
# 
# * Modeling Complex Artifacts: Muscle artifacts can have complex and variable patterns that might not be well captured by the linear assumptions of ICA. This can lead to incomplete removal or misidentification of artifact components.
# 
# * Source Mixing: EEG signals often arise from a mixture of neural, noise, and artifact sources. ICA can sometimes mix artifact components with neural sources, leading to challenges in isolating and removing artifacts without affecting neural signals.
# 
# * Lack of Gold Standard: There is no "gold standard" for verifying the accuracy of ICA-based artifact removal. Validation typically involves visual inspection and expert judgment, which can introduce subjectivity.
# 
# * Over- or Under-Removal: ICA might over-remove or under-remove artifact components depending on how well the assumptions match the actual data. Over-removal can distort the underlying neural activity, while under-removal leaves artifacts in the data.
# 
# * Varying Muscle Patterns: Muscle artifacts can vary between individuals and sessions, and ICA might not generalize well across different datasets or recording conditions.
# 
# * Component Validation: Identifying which extracted components are truly artifact-related and which are neural can be challenging. Manual validation requires expertise and can be time-consuming.
# 
# * Additional Preprocessing Steps: ICA is often combined with other preprocessing steps, such as band-pass filtering and rejection of contaminated components. These steps add complexity and might introduce additional artifacts.
# 
# **How it works**
# 
# ICA for removing muscle artifacts from EEG involves decomposing mixed EEG signals into statistically independent components. It identifies components that correspond to various sources, including muscle artifacts. By isolating and excluding these artifact-related components, ICA aims to enhance the quality of the EEG data.

# %%
"""
This script visualizes multi-channel data by plotting each channel with vertical offsets to separate them,
setting y-axis ticks and labels for clarity, and adding labels and a title,
facilitating a clear view of the data before Independent Component Analysis (ICA).
"""

n_channels = data_bp_filtered.shape[1]  # number of channels
plt.plot(data_bp_filtered + range(1000, 1000*n_channels+1000, 1000))
plt.yticks(range(1000, 1000*n_channels+1000, 1000), range(1, n_channels+1))
plt.ylabel('Channels, µV')
plt.xlabel('Sample')
plt.title("Data before ICA")
plt.show()

# %%
"""
The code first creates an instance of the FastICA class from the scikit-learn library,
specifying the number of components to be extracted as 8. The fit method is then called on the data_bp_filtered data, which performs the ICA decomposition and learns the unmixing matrix.
The transform method applies the learned unmixing matrix to the original data, resulting in the independent components stored in the components variable.
Next, the code plots the independent components using plt.plot,
with each component vertically offset by adding a constant value from the range(1, 1*n_channels+1, 1) sequence. This separation makes it easier to visualize the individual components.
"""

#pca=PCA(n_components=8)
ica=FastICA(n_components=8)

data_after_band_pass_filter_reshape = data_bp_filtered #np.transpose(data_bp_filtered)
#data_after_band_pass_filter_reshape = data_bp_filtered #np.transpose(data_bp_filtered)

#out_pca=pca.fit_transform(data_after_band_pass_filter_reshape)
out_ica=ica.fit_transform(data_after_band_pass_filter_reshape)

#fig, ax = plt.subplots(3,1,figsize=(10,8))
ica = FastICA(n_components=8)
ica.fit(data_after_band_pass_filter_reshape)
components = ica.transform(data_after_band_pass_filter_reshape)


offset = 5  # Adjust this value to increase or decrease the spacing
plt.figure(figsize=(12, 10))  # Increase figure size for better visibility
for i in range(8):
    plt.plot(components[:, i] + offset * i)

plt.yticks(range(0, offset * 8, offset), range(1, 9))
plt.ylabel('Components, µV')
plt.xlabel('Sample')
plt.title('Components data')
plt.show()

# %%
"""
After performing Independent Component Analysis (ICA) and visualizing the independent components, this script proceeds to remove all components except for the eye blink component (component 2) from the components matrix.
This is achieved by setting all other component values to zero using indexing operations.
The inverse_transform method of the ICA object is then called with the modified components matrix.
This method reconstructs the original data by applying the inverse of the unmixing matrix learned during the ICA decomposition.
The resulting data, stored in the restored variable, represents the original EEG data with the eye blink component removed.
Next, the code plots the restored data using plt.plot, with each channel vertically offset by adding a constant value from the range(1000, 1000*n_channels+1000, 1000) sequence.
This separation makes it easier to visualize the individual channels.
"""
# remove all components except for the eye blink component (component 2)
#components[:, 0] = 0
components[:, 1] = 0
components[:, 2] = 0
components[:, 3] = 0
components[:, 4] = 0
components[:, 5] = 0
components[:, 6] = 0
components[:, 7] = 0
# reconstruct EEG without blinks
restored = ica.inverse_transform(components)

plt.plot(restored + range(1000, 1000*n_channels+1000, 1000))
plt.yticks(range(1000, 1000*n_channels+1000, 1000), range(1, n_channels+1))
plt.ylabel('Channel')
plt.xlabel('Sample')
plt.title("Data after ICA")
plt.show()

channel_data = restored[:, channel]
plt.plot(channel_data, label=f'Channel {channel + 1} after ICA')
plt.plot(channel_data_bp_filtered, label=f'Channel {channel + 1} before ICA')
plt.ylabel('Amplitude')
plt.xlabel('Sample')
plt.title("Comparisaon data Data after ICA - Channel " + str(channel+1))
plt.legend()
plt.show()

# %%


# %% [markdown]
# As shown in the picture, the artifacts have been removed after applying ICA. However, you should be very careful, as this process might also remove some essential characteristics of the EEG signal along with the artifacts.   
# The detailed process for checking the correctness of artifact removal is described in the chapter.  
# 
# **Tasks**   
# 
#               # Experiment with different numbers of components for PCA and ICA.
# 
# **Expected Observations and Discussion**  
# 
#   By experimenting with different numbers of components for PCA and ICA on EEG data, you can gain insights into the strengths and limitations of these dimensionality reduction and source separation techniques. This will enhance their understanding of how to preprocess and analyze EEG signals effectively. Check how the number of channels affects the quality, do this method for 4 channels and 6 channels.
# 
# 
# 
# 
# 
# 

# %% [markdown]
# ## Wavelet decomposition
# 
# Wavelet analysis for EEG involves the application of wavelet transforms to EEG signals to extract valuable information from the time-frequency domain. The main advantage of using wavelet analysis for EEG lies in its ability to provide time-frequency representations, which is essential for studying the dynamic changes in brain activity over time.
# 
# **Positive**
# 
# *   Time-Frequency Localization: Wavelet transforms provide time-frequency localization. This means that they can reveal how the frequency content of a signal changes over time. Different wavelet scales allow you to zoom in on different frequency ranges while maintaining good time resolution. This is crucial for capturing rapid changes in EEG signal frequencies that correspond to various brain activities.
# 
# * Adaptability to Signal Changes: EEG signals are highly variable due to factors like different brain states (e.g., awake, asleep, under stress) and external stimuli. The adaptive nature of wavelet analysis allows it to effectively capture the changes in frequency and amplitude as they occur, making it suitable for non-stationary signals like EEG.
# 
# * Multi-Resolution Analysis: Wavelets provide a multi-resolution analysis, which means they can represent a signal at different scales. This is particularly useful for capturing both fine details and coarse trends in EEG signals. High-frequency components can be analyzed at finer scales, while low-frequency components can be analyzed at coarser scales.
# 
# * Artifact Removal: EEG signals can be contaminated with various artifacts, such as muscle activity or electrical interference. Wavelet decomposition can help separate these artifacts from the neural activity by highlighting differences in their time-frequency characteristics.
# 
# * Feature Extraction: For various EEG applications, such as brain-computer interfaces or medical diagnosis, relevant features need to be extracted from the signal. Wavelet coefficients at different scales can serve as features that capture different aspects of the signal's time-frequency structure.
# 
# **Negative**
# 
# *   Complexity and Interpretability: Wavelet decomposition can produce complex time-frequency representations, which might make interpretation challenging, especially for those who are not familiar with wavelet analysis. The interpretation of the wavelet coefficients and their relationship to specific neural processes might not be straightforward.
# 
# * Selection of Wavelet Basis: The choice of wavelet basis function can greatly impact the results of the decomposition. Different wavelets are suited for different types of signal characteristics. Selecting an appropriate wavelet basis requires domain knowledge and experimentation, and a poor choice can lead to misleading interpretations.
# 
# * Artifact Sensitivity: While wavelet analysis can help separate brain-related signals from artifacts, it's still sensitive to various types of artifacts, and it might not always provide a perfect separation. The effectiveness of artifact removal depends on the similarity between the artifact and the neural signal in terms of their time-frequency properties.
# 
# * Trade-off between Time and Frequency Resolution: Although wavelets provide a good compromise between time and frequency resolution, there's still a trade-off. Some EEG applications might require very high time or frequency resolution, which might not be optimally met by wavelet decomposition.
# 
# * Wavelet Parameters: The decomposition process involves selecting parameters like the number of decomposition levels and the scale of the wavelet. These parameters can affect the results and might need to be adjusted for different EEG datasets, making the analysis process somewhat subjective.
# 
# * Limited Capture of Complex Neural Processes: Some complex neural processes involve interactions across multiple frequency bands, and wavelet decomposition might struggle to fully capture these intricate relationships. More advanced techniques like higher-order spectral analysis or time-frequency coherence might be more suitable for such cases.
# 
# * Computationally Intensive: Wavelet decomposition can be computationally intensive, especially if high-resolution analyses are required. This can be a limitation when processing large amounts of EEG data.
# 
# **How it works**  
# 
# Wavelet decomposition for EEG analysis involves breaking down EEG signals into different frequency components while retaining information about their occurrence in time. This is achieved by convolving the EEG signal with a family of wavelet functions of varying scales. Each wavelet scale captures different frequency information, and the resulting coefficients highlight when and where these frequencies are present in the signal. This enables simultaneous exploration of time-varying frequency patterns in EEG, providing insights into dynamic brain activity over different scales of time and frequency.

# %%
"""
This script conducts a wavelet decomposition analysis on EEG (electroencephalogram) data.
Initially, it constructs a time array and scales the EEG data.
Then, it applies a continuous wavelet transform (CWT) using a Ricker wavelet to the scaled data.
This transformed data is visualized both before and after the wavelet decomposition process.
The first plot displays the raw EEG data, while the second plot illustrates the resulting wavelet decomposition, showcasing the power distribution across time and frequency bands.
"""

time = []
scaler_wave = []
count = 0

for _ in channel_data_bp_filtered:
    count = count+0.1
    time.append(count)

data_for_wave = np.array(channel_data_bp_filtered)

sc_X = StandardScaler()
data_for_wave = data_for_wave.reshape(-1,1)
scaler = sc_X.fit_transform(data_for_wave)  # apply standard scaler

for a in scaler:
    scaler_wave.append(a[0])

cwtmatr = signal.cwt(scaler_wave, signal.ricker, time)
cwtmatr_yflip = np.flipud(cwtmatr)

plt.plot(channel_data_bp_filtered)
plt.ylabel('EEG, µV')
plt.title("Before Wavelet ch, Channel " + str(channel + 1) )
plt.xlabel('Sample')
plt.show()

plt.imshow(
    cwtmatr_yflip,
    extent=[-70000, 70000, 100, 60000],
    cmap='PRGn', aspect='auto',
    vmax=abs(cwtmatr).max(),
    vmin=-abs(cwtmatr).max()
)
plt.ylabel('Power')
plt.title("After Wavelet decomposition, Channel " + str(channel + 1))
plt.xlabel('Sample')
plt.show()

# %% [markdown]
# **Tasks**   
# 
#     # Change Parameters in np. linspace.
#     This line creates the time vector t.
#     Changing the number of samples (200 in this case)
#     and the range (-1 to 1) will affect the generated signal and its frequency components.
#     For example:
# ```
# t = np.linspace(-1, 1, 100, endpoint=False)  # Fewer samples
# t = np.linspace(-1, 1, 400, endpoint=False)  # More samples
# ```
# **Expectation**
# 
# Changing the Number of Samples:
# 
# *   Increasing the number of samples will result in a higher resolution time vector, meaning the signal will have more points and appear smoother.
# *  Decreasing the number of samples will lower the resolution, making the signal more discrete and potentially missing finer details.
# *   A higher number of samples will also impact the frequency resolution in the signal processing steps, allowing for better frequency component analysis.
# 
# Changing the Range:
# 
# * Adjusting the range will affect the period over which the signal is generated.
# * A wider range (e.g., -2 to 2) will extend the time duration of the signal, potentially showing more cycles of the waveform.
# * A narrower range (e.g., -0.5 to 0.5) will shorten the time duration, capturing fewer cycles and potentially higher frequency components if the signal's frequency is high enough.
# 
# 
# **Tasks**   
# 
#       # This line performs the CWT using the Ricker wavelet.
#        Changing the wavelet function (e.g., signal.morlet, signal.haar)
#       or adjusting the time scales used will affect the transform.
#       
#       
# ```
# For example:
# cwtmatr = signal.cwt(scaler_wave, signal.morlet, np.arange(1, 31))  
# 
# ```
# **Expected Observations and Discussion**    
# CWT with Different Wavelets:
# 
# * The choice of wavelet will affect the appearance of the CWT plot. The Morlet wavelet will show smoother, oscillatory patterns, while the Haar wavelet will emphasize sharp changes. The Ricker wavelet will highlight transient features.  
# 
# CWT with Different Time Scales:  
# 
# *  The range of scales will affect the level of detail and frequency range displayed in the CWT plot. A narrow range will provide detailed analysis within a specific frequency band, while a wide range will offer a more complete frequency analysis.

# %% [markdown]


# %% [markdown]
# 
# ##Empirical Mode Decomposition  (EMD)
# EMD (Empirical Mode Decomposition) is a data analysis method used to explore the intrinsic oscillatory components present in EEG signals. EMD decomposes a complex EEG time series into simpler components called Intrinsic Mode Functions (IMFs) based on the data itself, without requiring any predefined basis functions.
# 
# **Positive**
# 
# 
# *   Adaptive Decomposition: EMD adaptively decomposes a signal into a set of intrinsic mode functions (IMFs) based on the local characteristics of the signal. This is beneficial for handling muscle artifacts, as they can vary in amplitude and frequency over time.
# 
# * Localized Frequency Decomposition: Muscle artifacts often manifest as high-frequency oscillations in EEG signals. EMD's ability to decompose signals into IMFs with distinct frequency ranges can help separate the muscle-related components from the neural activity of interest.
# 
# * Separation of Components: EMD can effectively separate the different components present in the EEG signal, including muscle artifacts, neural activity, and other noise sources. This separation can aid in the selective removal of unwanted components.
# 
# * Preservation of Temporal Features: EMD retains the temporal features of the original signal within each IMF, allowing it to capture both rapid and gradual changes caused by muscle artifacts.
# 
# * Non-Parametric Approach: EMD doesn't rely on predefined models or assumptions about the signal's structure, making it well-suited for handling the complex and non-stationary nature of muscle artifacts.
# 
# **Negative**
# 
# 
# *   Mode Mixing: EMD can suffer from mode mixing, where the intrinsic mode functions (IMFs) obtained may not cleanly separate the muscle artifacts from the neural signals. This can lead to incomplete or inaccurate artifact removal.
# 
# * Complexity of EEG Signals: EEG signals are complex and can include a variety of neural activities across different frequency ranges. EMD might not always effectively distinguish between muscle artifacts and neural components, especially when they overlap in frequency.
# 
# * Baseline Wander: EMD might not handle baseline wander effectively, which is a common low-frequency artifact in EEG signals. This can impact the accuracy of artifact removal.
# 
# * Subjectivity in Decomposition: EMD requires the selection of stopping criteria for the decomposition process, and these choices can be somewhat subjective. This subjectivity can lead to variations in results between different researchers or datasets.
# 
# * Artifact Variability: Muscle artifacts can vary widely in terms of amplitude, frequency, and time duration. EMD's adaptability might not guarantee consistent artifact removal across different EEG recordings or individuals.
# 
# * Computational Complexity: EMD can be computationally intensive, especially for longer EEG recordings. This can limit its practicality for real-time or large-scale applications.
# 
# **How it works**  
# EMD decomposes a signal into components called Intrinsic Mode Functions (IMFs) by identifying local maxima and minima, creating upper and lower envelopes via interpolation, and finding their average.
# 

# %% [markdown]
# ![emd.jpg](data:image/jpeg;base64,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)

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Create some data (for example, a sine wave with noise)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)

# Use scipy's find_peaks function
peaks, _ = find_peaks(y)
print (peaks)
# Plotting the data
plt.plot(x, y, label='Data')
plt.plot(x[peaks], y[peaks], 'ro', label='Peaks')
plt.legend()
plt.show()


# %%
"""
This Python script aims to detect and remove muscle artifacts from physiological data.
It starts by scaling the data and then applies the K-means clustering algorithm to identify artifacts.
Once an artifact is detected, it performs Empirical Mode Decomposition (EMD) to extract the artifact and replaces it with the EMD result.
Finally, it plots the original data alongside the cleaned data without artifacts.
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

EMD_data = channel_data_bp_filtered

for _ in EMD_data:
    count = count+0.1
    time.append(count)

just_for_graph = data_for_wave = EMD_data


sc_X = StandardScaler()
data_for_wave = data_for_wave.reshape(-1,1)
scaler = sc_X.fit_transform(data_for_wave)

scaler_wave = []
for a in scaler:
    scaler_wave.append(a[0])

kmeans = KMeans(
    n_clusters=2, init='k-means++', max_iter=3, tol=1000.001,
    verbose=10, random_state=1, n_init=10,  copy_x=True,
    algorithm='elkan') #algorithm= 'auto', 'full' or 'elkan
X = scaler
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

k_means_count = 0
shift_for_filter = 20
for_do_it_one_time = 0

plt.plot(scaler)
plt.plot(y_kmeans)
plt.legend(["EEG, µV","KNN to detect artefacts"])
plt.ylabel('EEG, µV')
plt.title("KNN to detact muscle artefacts, Channel " +str(channel + 1))
plt.xlabel('Sample')
plt.show()


for a in y_kmeans:
    k_means_count = k_means_count + 1
    if k_means_count>(shift_for_filter+shift_for_filter) and k_means_count<len(EMD_data)-shift_for_filter:
        if a == 1:
            if for_do_it_one_time == 0:
                plt.cla()
                for_do_it_one_time = 1
                data_for_EMD = EMD_data[(k_means_count-shift_for_filter - shift_for_filter):(k_means_count+shift_for_filter + shift_for_filter)]

                data_for_EMD = list(data_for_EMD)
                t = np.arange(0, len(data_for_EMD), 1)

                modes = data_for_EMD
                x = modes + t

                # Define signal
                t = np.linspace(0, 1, len(data_for_EMD))
                S = data_for_EMD
                eemd = EEMD()
                emd = eemd.EMD
                emd.extrema_detection="parabol"
                eIMFs = eemd.eemd(S, t)
                nIMFs = eIMFs.shape[0]
                data_for_replace = eIMFs[1]
                plt.plot(EMD_data)

                just_for_graph[(k_means_count-shift_for_filter - shift_for_filter):(k_means_count+shift_for_filter + shift_for_filter)] = data_for_replace
                plt.plot(just_for_graph)
                plt.ylabel('EEG, µV')
                plt.title("EMD to remove muscle artefacts, Channel " +str(channel + 1))
                plt.xlabel('Sample')

                plt.pause(0.1)
                plt.draw()

        else:
            for_do_it_one_time = 0
plt.plot(just_for_graph)
plt.ylabel('EEG, µV')
plt.title("EMD to remove muscle artefacts, Channel " +str(channel + 1))
plt.xlabel('Sample')

plt.pause(0.1)
plt.draw()

plt.plot(channel_data_bp_filtered)
plt.plot(just_for_graph)
plt.show()




# %% [markdown]
# ### Empirical Mode Decomposition (EMD) without library
# 
# In certain applications, particularly those involving real-time tasks, the execution time of a task holds paramount importance. Consequently, understanding the implementation of methods without relying on external libraries can prove to be invaluable. Such knowledge ultimately empowers you to optimize program execution time and achieve notable speed enhancements.Here we demonstrate how to Empirical Mode Decomposition (EMD) to remove muscle artefacts without library

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

global EMD_Y

# Assuming channel_data_bp_filtered is your signal data
# Replace with actual data for your use case
channel_data_bp_filtered = channel_data_bp_filtered #[:2000] #np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)

def detect_peak():
    # Detect peaks and valleys using scipy's find_peaks
    high_peaks, _ = find_peaks(channel_data_bp_filtered)
    low_peaks, _ = find_peaks(-channel_data_bp_filtered)

    # Interpolation for high peaks
    f_high = interp1d(high_peaks, channel_data_bp_filtered[high_peaks], kind='cubic', fill_value="extrapolate")
    x_new = np.arange(0, len(channel_data_bp_filtered), 1)
    y_high_inter = f_high(x_new)

    # Interpolation for low peaks
    f_low = interp1d(low_peaks, channel_data_bp_filtered[low_peaks], kind='cubic', fill_value="extrapolate")
    y_low_inter = f_low(x_new)

    # Plotting high and low peaks with their interpolations
    plt.plot (channel_data_bp_filtered)
    plt.plot(high_peaks, channel_data_bp_filtered[high_peaks], 'o', x_new, y_high_inter, '-')
    plt.plot(low_peaks, channel_data_bp_filtered[low_peaks], 'o', x_new, y_low_inter, '-')
    plt.title("Interpolation of high and low peaks")
    plt.legend(["Original Data", "Peak for high", "High interpolation", "Peak for low", "Low interpolation"])
    plt.ylabel('Signal Amplitude')
    plt.xlabel('Sample')
    plt.show()

    # Average graph after interpolation (EMD-like)
    EMD_Y = (y_high_inter + y_low_inter) / 2 # !!!
    print ("EMD_Y", EMD_Y)
    plt.title("Raw data and after interpolation")
    plt.ylabel('Signal Amplitude')
    plt.xlabel('Sample')
    plt.plot(EMD_Y - 10)
    plt.plot(channel_data_bp_filtered)
    plt.legend(["After interpolation", "Before interpolation"])
    plt.show()
    return EMD_Y

# Call the function to detect peaks and plot the results
EMD_Y = detect_peak()

print(EMD_Y)


# %% [markdown]
# **Tasks**  
# ```
#   # In the provided script, the lookahead parameter in the peakdetect function is a critical coefficient.  You can change this parameter and see the result.
#     Example
#   peaks = peakdetect(channel_data_bp_filtered, lookahead=5)
# ```
# **Expected Observations and Discussion**
# 
# This parameter controls how far the algorithm looks ahead to identify peaks and can significantly influence the detection of peaks in the data.  A larger lookahead value means the function will identify peaks over a broader range, which might be useful for identifying more prominent peaks. Conversely, a smaller lookahead value will make the function more sensitive to smaller fluctuations in the data.
# 
# More sensitive to minor fluctuations.
# May detect more peaks, including noise.
# Good for identifying fine details.
# Larger lookahead Values (e.g., 5-10).
# 
# Less sensitive to small fluctuations.
# Detects fewer, but more significant, peaks.
# Useful for smoothing out noise and focusing on major trends.
# Visualization and Understanding.

# %% [markdown]
# ##Canonical correlation analysis(CCA)
# Canonical Correlation Analysis (CCA) is a statistical technique used to explore and quantify the relationships between two sets of variables. In the context of EEG (electroencephalography) analysis, CCA is often applied to investigate the associations between two sets of EEG signals obtained from different brain regions or under different conditions.
# 
# **Positive**  
# *   Multivariate Analysis: EEG data often involves multiple channels capturing brain activity. CCA can effectively analyze the relationships between two sets of variables (e.g., EEG signals and stimuli) while considering their mutual interactions, making it suitable for EEG's multivariate nature.
# 
# * Feature Extraction: CCA can identify correlated components between EEG and external variables. These components can be used as informative features for classification or other downstream analyses, aiding in feature selection and reducing dimensionality.
# 
# * Artifact Removal: EEG data is susceptible to various artifacts (e.g., eye blinks, muscle activity). CCA can help in separating such artifacts from neural signals by identifying uncorrelated components, enhancing the quality of the extracted brain-related information.
# 
# * Functional Connectivity: CCA can uncover functional connectivity patterns between brain regions and external stimuli or tasks. This helps in understanding how different brain regions work together during specific cognitive processes.
# 
# * Non-linearity: CCA can be adapted to capture non-linear relationships between EEG data and external variables, enhancing its applicability to complex brain processes.
# 
# * Interpretability: CCA provides interpretable canonical variables that show how EEG signals and external variables are correlated. This can lead to insights into the cognitive mechanisms underlying the EEG responses.
# 
# * Customization: CCA allows for customization based on the specific research question or experimental design. You can tailor the analysis to focus on specific aspects of the data, which is particularly valuable in EEG studies.
# 
# * Reduced Dependency on Assumptions: CCA is relatively robust and doesn't require strong distributional assumptions, making it suitable for EEG data, which might not always follow strict statistical distributions.
# 
# **Negative**
# 
# *   Dimensionality: EEG data can involve a high number of channels and time points, leading to high-dimensional data. CCA's performance can degrade when dealing with such high dimensions, potentially resulting in overfitting or computational challenges.
# 
# * Complexity: CCA assumes a linear relationship between the two sets of variables. If the underlying relationship between EEG signals and external variables is non-linear, CCA might not capture the full complexity of the interactions.
# 
# * Interpretation: While CCA provides interpretable canonical variables, interpreting these variables can be challenging, especially in cases where the relationships between EEG and external variables are complex or not well understood.
# 
# * Data Preprocessing: CCA is sensitive to data preprocessing steps, such as normalization and artifact removal. If these preprocessing steps are not appropriately handled, they can impact the effectiveness of CCA and lead to unreliable results.
# 
# * Limited to Two Sets: CCA is designed for two sets of variables and might not be suitable for scenarios involving more than two sets of data or complex interactions among multiple variables.
# 
# * Overfitting: CCA can be prone to overfitting, especially when the number of observations is limited compared to the dimensionality of the data.
# 
# **How it works**  
# Canonical Correlation Analysis (CCA) is a statistical method used to uncover relationships between two sets of variables by finding linear combinations of each set that are maximally correlated. It aims to identify patterns of correlation between these sets. In the context of EEG analysis, CCA seeks to reveal meaningful associations between EEG signals and external variables (such as stimuli or tasks) by finding pairs of canonical variables that exhibit the highest possible correlation between the EEG and external data, allowing researchers to extract and interpret shared information.
# 

# %%
"""
This script demonstrates the application of Canonical Correlation Analysis (CCA) to analyze two sets of EEG data.
CCA is a statistical technique used to find associations between two multivariate sets of variables.
In this context, it's applied to find common patterns between EEG signals from two different channels.
First, it imports necessary libraries and prepares the EEG data.
Then, it initializes the CCA model with one canonical component.
After fitting the model to the data, it extracts the canonical components.
Finally, it visualizes the canonical components for each EEG channel, showing the shared patterns between them.
"""

import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA

#print ((eeg_data))
#print (eeg_data.shape)
#eeg_data = pd.DataFrame(eeg_data)
#EEG_data_1 = channel_data_bp_filtered # eeg_data.iloc[:, 0] # data_after_band_pass_filter.iloc[0]
#EEG_data_1 = channel_data[:, 0] #channel_data_bp_filtered #eeg_data.iloc[:, channel]
eeg_data = pd.read_excel("/content/drive/MyDrive/EEG_course/dataset/dataset.xlsx")

eeg_data = pd.DataFrame(eeg_data)
EEG_data_1 = eeg_data.iloc[:, 0]
# convert from Digital Value of Analog Digital converter (ADC) ADS1299 to microvolts µV
EEG_data_1 = round(1000000 * 4.5 * (EEG_data_1 / 16777215), 2)
#EEG_data_1 = np.array(EEG_data_1)
#EEG_data_1 = EEG_data_1.reshape(-1,1)

EEG_data_1 = EEG_data_1.to_numpy().reshape(-1, 1)

print (eeg_data)
EEG_data_2 = eeg_data.iloc[:, 1]
# convert from Digital Value of Analog Digital converter (ADC) ADS1299 to microvolts µV
EEG_data_2 = round(1000000 * 4.5 * (EEG_data_2 / 16777215), 2)

#EEG_data_2 = np.array(EEG_data_2)

EEG_data_2 = EEG_data_2.to_numpy().reshape(-1, 1)
#EEG_data_2 = EEG_data_2.reshape(-1,1)

cca = CCA(n_components=1)  # Choose the number of canonical components you want



# Fit the CCA model to the data
cca.fit(EEG_data_1, EEG_data_2)

# Get the canonical components for each set of EEG data
cca_components_1, cca_components_2 = cca.transform(EEG_data_1, EEG_data_2)

# Plot the canonical components
plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1)
plt.plot(cca_components_1, label='Canonical Components 1', color='green')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('EEG Channel 1 Canonical Component')
plt.legend()

plt.subplot(1, 4, 2)
plt.plot(cca_components_2, label='Canonical Components 2', color='orange')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('EEG Channel 2 Canonical Component')
plt.legend()


plt.subplot(1, 4, 3)
plt.plot(EEG_data_1,color='green')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('EEG Channel 1 Raw')
plt.legend()

plt.subplot(1, 4, 4)

plt.plot(EEG_data_2, color='orange')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('EEG Channel 2 Raw')
plt.show()







# %% [markdown]
# **Tasks**  
# ```
#       #Change the value of n_components and try to find a pattern
# ```
# **Expected Observations and Discussion**  
# This determines the number of canonical components to compute. By changing this parameter, the user can explore how the number of components affects the interpretation of the data. They can try different values and observe how it impacts the canonical components and their interpretation.
# 
# **Tasks**  
# EEG_data_1 and EEG_data_2: These variables represent the EEG data from two channels. The user can try using different EEG data from their dataset to see how different channels affect the canonical components and their interpretation.

# %% [markdown]
# **Evaluation**

# %% [markdown]
# Test

# %% [markdown]
# # Evaluation of artifact removal
# Artifact removal is crucial for obtaining clean and reliable EEG signals. Evaluation of artifact removal methods for EEG typically involves comparing the correlation between EEG signals and artifact-representing channels before and after applying removal algorithms. The effectiveness of a method is quantified by the extent to which it reduces this correlation across different artifact-inflicting tasks.

# %%
#A significant change in these values compared to the original data may indicate successful artifact removal, as muscle artifacts often introduce high-amplitude fluctuations
# arithmetic mean of array elements
mean_diff_after = np.mean(EMD_Y)
mean_diff_before = np.mean(channel_data_bp_filtered)
print ("np.mean, After", mean_diff_after, "Before", mean_diff_before)

# the standard deviation. It measures the amount of variation or dispersion in a dataset
std_diff_after = np.std(EMD_Y)
std_diff_before = np.std(channel_data_bp_filtered)
print ("np.std, After", std_diff_after, "Before", std_diff_before)

#Compute the correlation coefficient of the EEG data before and after artifact removal.
#A low correlation might suggest that the artifacts, which typically add noise, have been effectively reduced

corr_coef_before = np.corrcoef(EMD_Y)
corr_coef_after = np.corrcoef(channel_data_bp_filtered)

print ("np.corrcoef", "After", std_diff_after, "Before", std_diff_before)


# %% [markdown]
# This script is designed to evaluate the effectiveness of artifact removal in time-series data by performing a Fast Fourier Transform (FFT) analysis on signals before and after the removal process. The primary objective is to visualize and compare the frequency spectra of the data, allowing for a clear assessment of the impact of artifact removal on signal integrity.

# %%
import numpy as np
import matplotlib.pyplot as plt


sample_rate = 250  # Sampling rate in Hz
# Perform FFT
def furie (data,channel):
  fft_result = np.fft.fft(data)
  fft_magnitude = np.abs(fft_result) / len(data)  # Normalize magnitude

  # Compute the frequency axis
  N = len(data)
#  freqs = np.fft.fftfreq(N, d=1/sample_rate)
  plt.figure(figsize=(6, 4))

  # Time domain signal plot
  data_for_graph =  fft_magnitude[:N//2]
  plt.plot(data_for_graph)
  plt.xlim(0, 100)
  plt.title('Frequency Spectrum. ' + channel)
  plt.xlabel('Frequency [Hz]')
  plt.ylabel('Magnitude')
  plt.show()

furie(EMD_Y[1500:2500], "After artefact removed")
furie(channel_data_bp_filtered[1500:2500], "After artefact removed")



# %% [markdown]
# #SNR
# The SNR represents the ratio of the desired signal (brain activity) to the background noise (unwanted electrical activity). A higher SNR indicates that the brain signals are clearer and more distinguishable from noise, which is essential for accurate analysis.
# Poor SNR can lead to misinterpretation of EEG data. If noise is not properly separated from the signal, researchers may falsely conclude that certain neural responses are occurring when they are simply noise artifacts. Understanding and improving SNR is therefore vital to designing experiments that yield valid and reliable results.
# 
# The SNR before artefacts were removed is generally lower due to the presence of muscle artifacts, which can dominate the EEG signal.

# %%
import numpy as np
# Calculate power of the signal
P_signal_before = np.mean(channel_data_bp_filtered**2)
P_signal_after = np.mean(EMD_Y**2)

noise = pd.read_excel("/content/drive/MyDrive/EEG_course/dataset/Noise.xlsx")

# convert from Digital Value of Analog Digital converter (ADC) ADS1299 to microvolts µV
channel_data_noise = round(1000000 * 4.5 * (noise / 16777215), 2)
P_noise = np.mean(channel_data_noise**2)

# Calculate SNR
SNR_before = 10 * np.log10(P_signal_before / P_noise)
SNR_after = 10 * np.log10(P_signal_after / P_noise)

print(f"SNR: {SNR_before:.2f} dB")
print(f"SNR: {SNR_after:.2f} dB")

# %% [markdown]
# **The end of the chapter 4**
