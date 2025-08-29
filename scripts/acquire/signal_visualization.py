# acquire/signal_visualization.py
import json
import csv
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


class SignalProcessor:
    def __init__(self, sample_rate=250):
        self.sample_rate = sample_rate
        self.lowcut = 1
        self.highcut = 40
        
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a
    
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y
    
    def apply_bandpass_filter(self, data):
        filtered_data = self.butter_bandpass_filter(data, self.lowcut, self.highcut, self.sample_rate)
        return filtered_data


class SignalVisualization:
    def __init__(self, sample_rate=250):
        self.sample_rate = sample_rate
        self.processor = SignalProcessor(sample_rate)
        
    def load_data_from_csv(self, filename):
        data = {f'ch{i}': [] for i in range(1, 7)}
        timestamps = []
        
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if not row['timestamp'] or row['timestamp'].strip() == '':
                    continue
                    
                timestamps.append(float(row['timestamp']))
                for i in range(1, 7):
                    value = row.get(f'ch{i}', '0')
                    if value == '' or value is None:
                        value = '0.0'
                    data[f'ch{i}'].append(float(value))
        
        return data, timestamps
    
    def load_data_from_json(self, filename):
        with open(filename, 'r') as jsonfile:
            raw_data = json.load(jsonfile)
        
        data = {f'ch{i}': [] for i in range(1, 7)}
        timestamps = []
        
        for entry in raw_data:
            timestamps.append(entry['timestamp'])
            for i, val in enumerate(entry['ch1_6']):
                data[f'ch{i+1}'].append(val)
        
        return data, timestamps
    
    def plot_time_domain(self, data, timestamps=None, filter_data=True, save_plot=None):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        for i in range(6):
            row = i // 3
            col = i % 3
            
            ch_data = data[f'ch{i+1}']
            
            if filter_data and len(ch_data) > 10:
                filtered_data = self.processor.apply_bandpass_filter(ch_data)
            else:
                filtered_data = ch_data
            
            if timestamps and len(timestamps) == len(filtered_data):
                time_axis = [(t - timestamps[0]) for t in timestamps]
                axes[row, col].plot(time_axis, filtered_data, color='blue', linewidth=0.8)
                axes[row, col].set_xlabel('Time (s)')
            else:
                axes[row, col].plot(filtered_data, color='blue', linewidth=0.8)
                axes[row, col].set_xlabel('Sample')
            
            axes[row, col].set_ylabel('Amplitude (μV)')
            axes[row, col].set_title(f'Ch{i+1} - Time Domain')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.suptitle('EEG Signals - Time Domain (1-40 Hz)', fontsize=16)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(save_plot, dpi=300, bbox_inches='tight')
            print(f"Time domain plot saved to {save_plot}")
        
        plt.show()
    
    def plot_frequency_domain(self, data, save_plot=None):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        for i in range(6):
            row = i // 3
            col = i % 3
            
            ch_data = data[f'ch{i+1}']
            
            if len(ch_data) < 100:
                axes[row, col].text(0.5, 0.5, 'Insufficient data', 
                                  transform=axes[row, col].transAxes, ha='center')
                continue
            
            # Apply 1-40 Hz bandpass filter
            filtered_data = self.processor.apply_bandpass_filter(ch_data)
            
            # Compute power spectral density
            freqs, psd = signal.welch(filtered_data, self.sample_rate, 
                                    nperseg=min(512, len(filtered_data)//4))
            
            # Plot only 1-40 Hz range
            freq_mask = (freqs >= 1) & (freqs <= 40)
            freqs_plot = freqs[freq_mask]
            psd_plot = psd[freq_mask]
            
            axes[row, col].semilogy(freqs_plot, psd_plot, color='red', linewidth=1)
            axes[row, col].set_xlabel('Frequency (Hz)')
            axes[row, col].set_ylabel('Power (μV²/Hz)')
            axes[row, col].set_title(f'Ch{i+1} - Frequency Domain')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_xlim(1, 40)
            
            # Mark frequency bands
            axes[row, col].axvline(8, color='green', linestyle='--', alpha=0.5, label='α')
            axes[row, col].axvline(13, color='orange', linestyle='--', alpha=0.5, label='β')
            axes[row, col].axvline(30, color='purple', linestyle='--', alpha=0.5, label='γ')
        
        plt.suptitle('EEG Signals - Frequency Domain (1-40 Hz)', fontsize=16)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(save_plot, dpi=300, bbox_inches='tight')
            print(f"Frequency domain plot saved to {save_plot}")
        
        plt.show()
    
    def visualize_data(self, data, timestamps=None, save_plots=None):
        # Plot time domain
        time_save = f"{save_plots}_time.png" if save_plots else None
        self.plot_time_domain(data, timestamps, filter_data=True, save_plot=time_save)
        
        # Plot frequency domain
        freq_save = f"{save_plots}_freq.png" if save_plots else None
        self.plot_frequency_domain(data, save_plot=freq_save)


if __name__ == "__main__":
    visualizer = SignalVisualization()
    
    try:
        data, timestamps = visualizer.load_data_from_csv("eeg_data_20250829_100618.csv")
        visualizer.visualize_data(data, timestamps, save_plots="eeg_analysis")
        
    except FileNotFoundError:
        print("Data file not found. Please record data first using data_storage.py")
