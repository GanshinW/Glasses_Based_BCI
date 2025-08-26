import json
import csv
import time
import threading
from datetime import datetime
from signal_acquisition import SignalAcquisition


class DataStorage:
    def __init__(self, filename_prefix="eeg_data"):
        self.filename_prefix = filename_prefix
        self.acquisition = SignalAcquisition()
        self.data_buffer = []
        self.is_recording = False
        self.recording_thread = None
        
    def start_recording(self, duration=None):
        if self.is_recording:
            print("Recording already in progress")
            return
        
        self.is_recording = True
        self.data_buffer = []
        self.acquisition.start_acquisition()
        
        self.recording_thread = threading.Thread(
            target=self._record_data, 
            args=(duration,)
        )
        self.recording_thread.start()
        print("Recording started")
    
    def stop_recording(self):
        if not self.is_recording:
            print("No recording in progress")
            return
        
        self.is_recording = False
        self.acquisition.stop_acquisition()
        
        if self.recording_thread:
            self.recording_thread.join()
        
        print("Recording stopped")
    
    def _record_data(self, duration):
        start_time = time.time()
        
        while self.is_recording:
            if duration and (time.time() - start_time) >= duration:
                break
            
            data1, data2 = self.acquisition.read_data()
            
            if data1 is not None and data2 is not None:
                timestamp = time.time()
                combined_data = {
                    'timestamp': timestamp,
                    'ch1_6': data1[:6]  # Only store first 6 channels from first ADS1299
                }
                self.data_buffer.append(combined_data)
            
            time.sleep(0.001)
        
        self.is_recording = False
    
    def save_to_csv(self, filename=None):
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.filename_prefix}_{timestamp}.csv"
        
        if not self.data_buffer:
            print("No data to save")
            return
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp'] + [f'ch{i}' for i in range(1, 7)]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for entry in self.data_buffer:
                row = {'timestamp': entry['timestamp']}
                for i, val in enumerate(entry['ch1_6']):
                    row[f'ch{i+1}'] = val
                writer.writerow(row)
        
        print(f"Data saved to {filename}")
        return filename
    
    def save_to_json(self, filename=None):
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.filename_prefix}_{timestamp}.json"
        
        if not self.data_buffer:
            print("No data to save")
            return
        
        with open(filename, 'w') as jsonfile:
            json.dump(self.data_buffer, jsonfile, indent=2)
        
        print(f"Data saved to {filename}")
        return filename
    
    def get_data_buffer(self):
        return self.data_buffer.copy()
    
    def clear_buffer(self):
        self.data_buffer = []
        print("Data buffer cleared")


if __name__ == "__main__":
    storage = DataStorage()
    
    try:
        # Record for 10 seconds
        storage.start_recording(duration=10)
        
        # Wait for recording to complete
        while storage.is_recording:
            time.sleep(0.1)
        
        # Save data
        storage.save_to_csv()
        storage.save_to_json()
        
    except KeyboardInterrupt:
        print("Recording interrupted")
        storage.stop_recording()
        storage.save_to_csv()
        storage.save_to_json()
