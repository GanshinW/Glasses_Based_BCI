#!/usr/bin/env python3
"""
Real-time Prediction System
Uses sliding window approach with unified preprocessing
Demonstrates how training data window size matches real-time inference
"""

import numpy as np
import time
from collections import deque
from signal_acquisition import SignalAcquisition
from unified_preprocessor import UnifiedPreprocessor


class RealTimePredictor:
    def __init__(self, model_path=None, window_length_sec=2.0, sampling_rate=250):
        self.window_length_sec = window_length_sec
        self.sampling_rate = sampling_rate
        self.window_samples = int(window_length_sec * sampling_rate)
        
        # Signal acquisition and preprocessing
        self.acquisition = SignalAcquisition()
        self.preprocessor = UnifiedPreprocessor(sampling_rate)
        
        # Sliding window buffer
        self.signal_buffer = deque(maxlen=self.window_samples)
        self.n_channels = 6
        
        # Model placeholder
        self.model = None  # Load your trained model here
        self.class_names = []
        
        # Prediction parameters
        self.prediction_interval = 0.5  # seconds between predictions
        self.min_confidence = 0.7
        
    def load_model(self, model_path, class_names):
        """
        Load trained model
        Args:
            model_path: path to saved model
            class_names: list of class names
        """
        # Placeholder for model loading
        # self.model = load_model(model_path)
        self.class_names = class_names
        print(f"Model loaded: {model_path}")
        print(f"Classes: {class_names}")
        
    def start_prediction(self):
        """Start real-time prediction"""
        if self.model is None:
            print("Warning: No model loaded, using dummy predictions")
            
        print("Starting real-time prediction...")
        print(f"Window length: {self.window_length_sec}s")
        print(f"Buffer size: {self.window_samples} samples")
        
        self.acquisition.start_acquisition()
        
        try:
            last_prediction_time = 0
            
            while True:
                # Read new data
                data1, data2 = self.acquisition.read_data()
                
                if data1 is not None:
                    # Add to buffer (use first 6 channels)
                    self.signal_buffer.append(data1[:self.n_channels])
                    
                    # Check if we have enough data and enough time has passed
                    current_time = time.time()
                    if (len(self.signal_buffer) == self.window_samples and 
                        current_time - last_prediction_time >= self.prediction_interval):
                        
                        # Make prediction
                        prediction = self._predict_current_window()
                        if prediction is not None:
                            self._display_prediction(prediction)
                            
                        last_prediction_time = current_time
                        
                time.sleep(0.001)  # Small delay to prevent 100% CPU usage
                
        except KeyboardInterrupt:
            print("\nStopping real-time prediction...")
        finally:
            self.acquisition.stop_acquisition()
            
    def _predict_current_window(self):
        """
        Make prediction on current window
        Returns:
            prediction dictionary or None
        """
        # Convert buffer to numpy array
        window_data = np.array(list(self.signal_buffer))  # shape: (samples, channels)
        window_data = window_data.T  # shape: (channels, samples)
        
        # Apply same preprocessing as training
        try: