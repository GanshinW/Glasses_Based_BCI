# scripts/realtime_bci_predictor.py#!/usr/bin/env python3
"""
Real-time BCI prediction system with EEG signal acquisition, visualization and emotion/intent classification
"""

import numpy as np
import torch
import torch.nn as nn
import time
import threading
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys

try:
    from acquire.signal_acquisition import SignalAcquisition
    from models.multimodal_model import MultiModalNet
    from utils.features import extract_time_domain, extract_freq_domain, extract_timefreq_images
    from utils.preprocess import bandpass_filter, normalize_data
    from config import Config
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you are running from the scripts/ directory")
    sys.exit(1)


class RealTimeBCIPredictor:
    def __init__(self, emotion_model_path=None, intent_model_path=None):
        # Basic parameters
        self.sampling_rate = 250
        self.n_channels = 6
        self.window_length_sec = 2.0
        self.window_samples = int(self.window_length_sec * self.sampling_rate)
        self.prediction_interval = 0.5
        
        # Signal buffers
        self.signal_buffer = deque(maxlen=self.window_samples)
        self.time_stamps = deque(maxlen=self.window_samples)
        
        # Visualization buffer (keep 5 seconds for display)
        self.vis_length = int(5 * self.sampling_rate)
        self.vis_buffer = [deque(maxlen=self.vis_length) for _ in range(self.n_channels)]
        self.vis_time = deque(maxlen=self.vis_length)
        
        # Signal acquisition
        self.acquisition = SignalAcquisition()
        self.is_running = False
        
        # Models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotion_model = None
        self.intent_model = None
        self.emotion_model_path = emotion_model_path
        self.intent_model_path = intent_model_path
        
        # Class labels
        self.emotion_classes = ['neutral', 'Positive', 'Negative']
        self.intent_classes = ['Baseline', 'GazeLeft', 'GazeRight','JawClench']
        
        # Prediction results
        self.current_emotion = "Unknown"
        self.current_intent = "Unknown"
        self.emotion_confidence = 0.0
        self.intent_confidence = 0.0
        
        # Threading control
        self.data_thread = None
        self.prediction_thread = None
        
        # Load models
        if emotion_model_path and os.path.exists(emotion_model_path):
            self.load_emotion_model(emotion_model_path)
        if intent_model_path and os.path.exists(intent_model_path):
            self.load_intent_model(intent_model_path)
    
    def load_emotion_model(self, model_path):
        """Load pre-trained emotion classification model"""
        try:
            config = Config()
            self.emotion_model = MultiModalNet(
                n_channels=self.n_channels,
                n_samples=self.window_samples,
                n_bands=len(config.bands),
                img_out_dim=config.img_out_dim,
                hidden_dim=config.hidden_dim,
                n_classes=len(self.emotion_classes),
                use_img=config.use_img
            ).to(self.device)
            
            self.emotion_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.emotion_model.eval()
            print(f"Emotion model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load emotion model: {e}")
    
    def load_intent_model(self, model_path):
        """Load pre-trained intent classification model"""
        try:
            config = Config()
            self.intent_model = MultiModalNet(
                n_channels=self.n_channels,
                n_samples=self.window_samples,
                n_bands=len(config.bands),
                img_out_dim=config.img_out_dim,
                hidden_dim=config.hidden_dim,
                n_classes=len(self.intent_classes),
                use_img=config.use_img
            ).to(self.device)
            
            self.intent_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.intent_model.eval()
            print(f"Intent model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load intent model: {e}")
    
    def preprocess_signal(self, signal_data):
        """Preprocess EEG signal using utils/preprocess.py functions"""
        # signal_data shape: (samples, channels) -> need (channels, samples)
        signal_array = np.array(signal_data).T  # Convert to (channels, samples)
        
        # Apply bandpass filter (1-40 Hz)
        filtered_data = bandpass_filter(signal_array, lowcut=1.0, highcut=40.0, 
                                       fs=self.sampling_rate, order=4)
        
        # Apply z-score normalization
        normalized_data = normalize_data(filtered_data)
        
        return normalized_data
    
    def extract_features(self, signal_data):
        """Extract multimodal features from preprocessed signal"""
        try:
            config = Config()
            
            # Time domain features
            X_time = extract_time_domain(signal_data[np.newaxis, :, :], self.sampling_rate)
            
            # Frequency domain features
            X_freq = extract_freq_domain(X_time, self.sampling_rate, config.bands)
            
            # Time-frequency images (optional)
            X_img = None
            if config.use_img:
                X_img = extract_timefreq_images(X_time, self.sampling_rate)
            
            return X_time, X_freq, X_img
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None, None, None
    
    def predict_emotion(self, X_time, X_freq, X_img):
        """Predict emotion from features"""
        if self.emotion_model is None:
            return "No Model", 0.0
        
        try:
            with torch.no_grad():
                time_tensor = torch.tensor(X_time, dtype=torch.float32).to(self.device)
                freq_tensor = torch.tensor(X_freq, dtype=torch.float32).to(self.device)
                img_tensor = torch.tensor(X_img, dtype=torch.float32).to(self.device) if X_img is not None else None
                
                outputs = self.emotion_model(time_tensor, freq_tensor, img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                return self.emotion_classes[predicted.item()], confidence.item()
        except Exception as e:
            print(f"Emotion prediction error: {e}")
            return "Error", 0.0
    
    def predict_intent(self, X_time, X_freq, X_img):
        """Predict intent from features"""
        if self.intent_model is None:
            return "No Model", 0.0
        
        try:
            with torch.no_grad():
                time_tensor = torch.tensor(X_time, dtype=torch.float32).to(self.device)
                freq_tensor = torch.tensor(X_freq, dtype=torch.float32).to(self.device)
                img_tensor = torch.tensor(X_img, dtype=torch.float32).to(self.device) if X_img is not None else None
                
                outputs = self.intent_model(time_tensor, freq_tensor, img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                return self.intent_classes[predicted.item()], confidence.item()
        except Exception as e:
            print(f"Intent prediction error: {e}")
            return "Error", 0.0
    
    def data_acquisition_loop(self):
        """Main data acquisition loop"""
        print("Starting data acquisition...")
        self.acquisition.start_acquisition()
        
        try:
            while self.is_running:
                # Read new data from acquisition system
                data1, data2 = self.acquisition.read_data()
                
                if data1 is not None:
                    # Use first 6 channels from data1
                    current_time = time.time()
                    channel_data = data1[:self.n_channels]
                    
                    # Add to buffers
                    self.signal_buffer.append(channel_data)
                    self.time_stamps.append(current_time)
                    
                    # Update visualization buffer
                    for i in range(self.n_channels):
                        self.vis_buffer[i].append(channel_data[i])
                    self.vis_time.append(current_time)
                
                time.sleep(0.001)  # Small delay to prevent 100% CPU usage
                
        except Exception as e:
            print(f"Data acquisition error: {e}")
        finally:
            self.acquisition.stop_acquisition()
    
    def prediction_loop(self):
        """Main prediction loop"""
        print("Starting prediction loop...")
        last_prediction_time = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if we have enough data and enough time has passed
                if (len(self.signal_buffer) >= self.window_samples and 
                    current_time - last_prediction_time >= self.prediction_interval):
                    
                    # Convert buffer to numpy array for processing
                    window_data = np.array(list(self.signal_buffer))  # (samples, channels)
                    window_data = window_data.T  # (channels, samples)
                    
                    # Preprocess signal
                    preprocessed_data = self.preprocess_signal(window_data)
                    
                    # Extract features
                    X_time, X_freq, X_img = self.extract_features(preprocessed_data)
                    
                    if X_time is not None:
                        # Make predictions
                        emotion, emotion_conf = self.predict_emotion(X_time, X_freq, X_img)
                        intent, intent_conf = self.predict_intent(X_time, X_freq, X_img)
                        
                        # Update results
                        self.current_emotion = emotion
                        self.current_intent = intent
                        self.emotion_confidence = emotion_conf
                        self.intent_confidence = intent_conf
                        
                        # Print results
                        print(f"Emotion: {emotion} ({emotion_conf:.2f}) | Intent: {intent} ({intent_conf:.2f})")
                    
                    last_prediction_time = current_time
                
                time.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                print(f"Prediction error: {e}")
                time.sleep(1)
    
    def start_prediction(self):
        """Start real-time prediction system"""
        if self.is_running:
            print("System is already running!")
            return
        
        print("Starting real-time BCI prediction system...")
        self.is_running = True
        
        # Start data acquisition thread
        self.data_thread = threading.Thread(target=self.data_acquisition_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        # Start prediction thread
        self.prediction_thread = threading.Thread(target=self.prediction_loop)
        self.prediction_thread.daemon = True
        self.prediction_thread.start()
    
    def stop_prediction(self):
        """Stop real-time prediction system"""
        print("Stopping real-time prediction system...")
        self.is_running = False
        
        # Wait for threads to finish
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=2)
        if self.prediction_thread and self.prediction_thread.is_alive():
            self.prediction_thread.join(timeout=2)
    
    def create_visualization_gui(self):
        """Create GUI with real-time signal visualization and prediction results"""
        
        class BCIVisualizationGUI:
            def __init__(self, predictor):
                self.predictor = predictor
                self.root = tk.Tk()
                self.root.title("Real-time BCI Prediction System")
                self.root.geometry("1200x800")
                
                # Create main frames
                self.setup_gui()
                
                # Setup matplotlib
                self.setup_plots()
                
                # Start animation
                self.ani = animation.FuncAnimation(
                    self.fig, self.update_plots, interval=100, blit=False
                )
            
            def setup_gui(self):
                # Control frame
                control_frame = ttk.Frame(self.root)
                control_frame.pack(fill=tk.X, padx=10, pady=5)
                
                # Start/Stop buttons
                self.start_btn = ttk.Button(control_frame, text="Start", command=self.start_system)
                self.start_btn.pack(side=tk.LEFT, padx=5)
                
                self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_system)
                self.stop_btn.pack(side=tk.LEFT, padx=5)
                
                # Status labels
                self.status_label = ttk.Label(control_frame, text="Status: Stopped")
                self.status_label.pack(side=tk.LEFT, padx=20)
                
                # Prediction results frame
                pred_frame = ttk.LabelFrame(self.root, text="Prediction Results")
                pred_frame.pack(fill=tk.X, padx=10, pady=5)
                
                # Emotion result
                emotion_frame = ttk.Frame(pred_frame)
                emotion_frame.pack(fill=tk.X, pady=2)
                ttk.Label(emotion_frame, text="Current Emotion:").pack(side=tk.LEFT)
                self.emotion_label = ttk.Label(emotion_frame, text="Unknown", font=('Arial', 12, 'bold'))
                self.emotion_label.pack(side=tk.LEFT, padx=10)
                self.emotion_conf_label = ttk.Label(emotion_frame, text="(0.00)")
                self.emotion_conf_label.pack(side=tk.LEFT)
                
                # Intent result
                intent_frame = ttk.Frame(pred_frame)
                intent_frame.pack(fill=tk.X, pady=2)
                ttk.Label(intent_frame, text="Current Intent:").pack(side=tk.LEFT)
                self.intent_label = ttk.Label(intent_frame, text="Unknown", font=('Arial', 12, 'bold'))
                self.intent_label.pack(side=tk.LEFT, padx=10)
                self.intent_conf_label = ttk.Label(intent_frame, text="(0.00)")
                self.intent_conf_label.pack(side=tk.LEFT)
            
            def setup_plots(self):
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                
                # Create matplotlib figure
                self.fig, self.axes = plt.subplots(3, 2, figsize=(12, 8))
                self.fig.suptitle('Real-time EEG Signals (6 Channels)', fontsize=14)
                
                # Channel names
                channel_names = ['F7', 'F8', 'T3', 'T4', 'FT7', 'FT8']
                
                # Initialize plots
                self.lines = []
                for i in range(6):
                    row = i // 2
                    col = i % 2
                    ax = self.axes[row, col]
                    line, = ax.plot([], [], 'b-', linewidth=1)
                    self.lines.append(line)
                    
                    ax.set_title(f'Channel {channel_names[i]}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Amplitude (μV)')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, 5)
                    ax.set_ylim(-100, 100)
                
                plt.tight_layout()
                
                # Embed plot in tkinter
                self.canvas = FigureCanvasTkAgg(self.fig, self.root)
                self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            def update_plots(self, frame):
                """Update real-time plots"""
                if not self.predictor.vis_time:
                    return self.lines
                
                # Get current time reference
                current_time = time.time()
                time_data = np.array(self.predictor.vis_time) - current_time
                
                # Update each channel
                for i in range(6):
                    if len(self.predictor.vis_buffer[i]) > 0:
                        signal_data = np.array(self.predictor.vis_buffer[i])
                        self.lines[i].set_data(time_data, signal_data)
                
                # Update prediction labels
                self.emotion_label.config(text=self.predictor.current_emotion)
                self.emotion_conf_label.config(text=f"({self.predictor.emotion_confidence:.2f})")
                self.intent_label.config(text=self.predictor.current_intent)
                self.intent_conf_label.config(text=f"({self.predictor.intent_confidence:.2f})")
                
                return self.lines
            
            def start_system(self):
                self.predictor.start_prediction()
                self.status_label.config(text="Status: Running")
                self.start_btn.config(state="disabled")
                self.stop_btn.config(state="normal")
            
            def stop_system(self):
                self.predictor.stop_prediction()
                self.status_label.config(text="Status: Stopped")
                self.start_btn.config(state="normal")
                self.stop_btn.config(state="disabled")
            
            def run(self):
                self.root.mainloop()
        
        # Create and run GUI
        gui = BCIVisualizationGUI(self)
        return gui


def main():
    """
    Main function to run the real-time BCI prediction system
    
    Usage:
    1. Place your trained .pth model files in the scripts/ directory:
       - emotion_model.pth (for emotion classification: Calm/Positive/Negative)
       - intent_model.pth (for intent classification: Baseline/EyeMove/JawClench)
    
    2. Make sure you're in the scripts/ directory when running:
       cd /path/to/your/project/scripts
       python realtime_bci_predictor.py
    
    3. The GUI will show:
       - 6 channel real-time EEG signals (F7, F8, T3, T4, FT7, FT8)
       - Current emotion prediction with confidence
       - Current intent prediction with confidence
       - Start/Stop controls
    
    Requirements:
    - scipy (for signal filtering)
    - matplotlib (for real-time plots)
    - tkinter (usually comes with Python)
    - torch, numpy
    - Your project's acquire/, models/, utils/ modules
    """
    
    # Model paths - update these with your actual model files
    emotion_model_path = "emotion_model.pth"  # Path to emotion classification model
    intent_model_path = "intent_model.pth"    # Path to intent classification model
    
    # Check if model files exist
    if not os.path.exists(emotion_model_path):
        print(f"Warning: Emotion model not found at {emotion_model_path}")
        emotion_model_path = None
    
    if not os.path.exists(intent_model_path):
        print(f"Warning: Intent model not found at {intent_model_path}")
        intent_model_path = None
    
    if emotion_model_path is None and intent_model_path is None:
        print("No model files found. Please place your .pth files in the scripts/ directory.")
        print("The system will still run but show 'No Model' for predictions.")
    
    # Create predictor
    predictor = RealTimeBCIPredictor(emotion_model_path, intent_model_path)
    
    # Create and run GUI
    gui = predictor.create_visualization_gui()
    
    try:
        print("Starting Real-time BCI Prediction System...")
        print("Close the GUI window to exit.")
        gui.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        predictor.stop_prediction()


if __name__ == "__main__":
    main()