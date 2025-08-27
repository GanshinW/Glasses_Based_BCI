# acquire/intent_acquisition.py
"""
Intent Detection Data Acquisition Script
Manual control - each trial saved as individual CSV
Shared JSON for session environment info
"""

import json
import csv
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import numpy as np
import os
from signal_acquisition import SignalAcquisition


class IntentAcquisitionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Intent Detection Data Acquisition")
        self.root.geometry("800x600")
        
        # Create data directories
        self.data_dir = os.path.join("data", "intent")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Data acquisition
        self.acquisition = SignalAcquisition()
        self.is_recording = False
        self.trial_count = 0
        self.participant_id = "P001"
        
        # Experiment parameters
        self.actions = ["baseline", "jaw_clench", "gaze_left", "gaze_right"]
        self.action_names = {
            "baseline": "Baseline",
            "jaw_clench": "Jaw Clench", 
            "gaze_left": "Gaze Left",
            "gaze_right": "Gaze Right"
        }
        
        # Trial timing (seconds)
        self.baseline_duration = 3
        self.action_duration = 3
        self.recovery_duration = 2
        self.trial_duration = self.baseline_duration + self.action_duration + self.recovery_duration
        
        # Current trial state
        self.trial_start_time = 0
        self.current_phase = "idle"
        self.current_action = ""
        self.current_trial_data = []
        
        # Setup GUI
        self._setup_gui()
        self._load_session_info()
        
    def _setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Intent Detection Data Acquisition", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Participant info
        ttk.Label(main_frame, text="Participant ID:").grid(row=1, column=0, sticky=tk.W)
        self.participant_entry = ttk.Entry(main_frame)
        self.participant_entry.insert(0, self.participant_id)
        self.participant_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))
        
        # Trial count info
        ttk.Label(main_frame, text="Completed Trials:").grid(row=2, column=0, sticky=tk.W)
        self.trial_count_var = tk.StringVar(value="0")
        ttk.Label(main_frame, textvariable=self.trial_count_var).grid(row=2, column=1, sticky=tk.W)
        
        # Current action display
        self.action_frame = ttk.LabelFrame(main_frame, text="Current Trial", padding="20")
        self.action_frame.grid(row=3, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))
        
        self.action_label = ttk.Label(self.action_frame, text="Click 'Next Trial' to begin", 
                                     font=("Arial", 24, "bold"))
        self.action_label.pack()
        
        self.phase_label = ttk.Label(self.action_frame, text="", 
                                    font=("Arial", 14))
        self.phase_label.pack()
        
        self.timer_label = ttk.Label(self.action_frame, text="", 
                                    font=("Arial", 12))
        self.timer_label.pack()
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        self.next_trial_button = ttk.Button(button_frame, text="Next Trial", 
                                           command=self.start_next_trial)
        self.next_trial_button.pack(side=tk.LEFT, padx=5)
        
        self.quit_button = ttk.Button(button_frame, text="Quit", 
                                     command=self.quit_application)
        self.quit_button.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Click 'Next Trial' to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def _load_session_info(self):
        """Load or create session info"""
        self.session_file = os.path.join(self.data_dir, "session_info.json")
        
        if os.path.exists(self.session_file):
            with open(self.session_file, 'r') as f:
                session_info = json.load(f)
                self.trial_count = session_info.get('total_trials', 0)
                self.trial_count_var.set(str(self.trial_count))
                
    def _update_session_info(self):
        """Update session info JSON"""
        session_info = {
            'task_type': 'intent_detection',
            'participant_id': self.participant_id,
            'last_updated': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'sampling_rate': 250,
            'electrode_positions': ['F7', 'FT7', 'T7', 'F8', 'FT8', 'T8'],
            'trial_structure': {
                'baseline_duration': self.baseline_duration,
                'action_duration': self.action_duration,
                'recovery_duration': self.recovery_duration,
                'total_duration': self.trial_duration
            },
            'total_trials': self.trial_count,
            'action_types': self.actions
        }
        
        with open(self.session_file, 'w') as f:
            json.dump(session_info, f, indent=2)
            
    def start_next_trial(self):
        self.participant_id = self.participant_entry.get().strip()
        if not self.participant_id:
            messagebox.showerror("Error", "Please enter participant ID")
            return
            
        # Generate next action randomly
        self.current_action = np.random.choice(self.actions)
        self.trial_start_time = time.time()
        self.current_phase = "baseline"
        self.current_trial_data = []
        
        # Disable button during trial
        self.next_trial_button.config(state=tk.DISABLED)
        
        # Start acquisition if not already started
        if not self.is_recording:
            self.acquisition.start_acquisition()
            self.is_recording = True
            
        self.status_var.set("Trial in progress...")
        
        # Start baseline phase
        self._update_display()
        self._record_trial_data()
        
        # Schedule phases
        self.root.after(int(self.baseline_duration * 1000), self._start_action_phase)
        
    def _start_action_phase(self):
        self.current_phase = "action"
        self._update_display()
        self.root.after(int(self.action_duration * 1000), self._start_recovery_phase)
        
    def _start_recovery_phase(self):
        self.current_phase = "recovery"
        self._update_display()
        self.root.after(int(self.recovery_duration * 1000), self._finish_trial)
        
    def _finish_trial(self):
        # Save this trial
        self._save_current_trial()
        
        # Update counters
        self.trial_count += 1
        self.trial_count_var.set(str(self.trial_count))
        self._update_session_info()
        
        # Re-enable button for next trial
        self.next_trial_button.config(state=tk.NORMAL)
        
        self.action_label.config(text="Trial completed")
        self.phase_label.config(text="Click 'Next Trial' for next trial")
        self.timer_label.config(text="")
        self.status_var.set("Ready for next trial")
        
    def _update_display(self):
        if self.current_phase == "baseline":
            self.action_label.config(text="Get Ready")
            self.phase_label.config(text="Baseline - Stay relaxed")
        elif self.current_phase == "action":
            self.action_label.config(text=self.action_names[self.current_action])
            self.phase_label.config(text="Perform action now")
        elif self.current_phase == "recovery":
            self.action_label.config(text="Relax")
            self.phase_label.config(text="Recovery - Return to baseline")
            
        self._update_timer()
        
    def _update_timer(self):
        if self.current_phase == "idle":
            return
            
        elapsed = time.time() - self.trial_start_time
        
        if self.current_phase == "baseline":
            remaining = self.baseline_duration - elapsed
        elif self.current_phase == "action":
            remaining = (self.baseline_duration + self.action_duration) - elapsed
        else:  # recovery
            remaining = self.trial_duration - elapsed
            
        self.timer_label.config(text=f"Time remaining: {remaining:.1f}s")
        
        if remaining > 0:
            self.root.after(100, self._update_timer)
            
    def _record_trial_data(self):
        """Record data for current trial in background thread"""
        def record_data():
            start_time = time.time()
            while (time.time() - start_time) < self.trial_duration:
                data1, data2 = self.acquisition.read_data()
                if data1 is not None:
                    current_time = time.time()
                    phase_time = current_time - self.trial_start_time
                    
                    # Determine current phase based on timing
                    if phase_time < self.baseline_duration:
                        phase = "baseline"
                    elif phase_time < (self.baseline_duration + self.action_duration):
                        phase = "action"
                    else:
                        phase = "recovery"
                    
                    sample = {
                        'timestamp': current_time,
                        'phase_time': phase_time,
                        'phase': phase,
                        'ch1': data1[0], 'ch2': data1[1], 'ch3': data1[2],
                        'ch4': data1[3], 'ch5': data1[4], 'ch6': data1[5]
                    }
                    self.current_trial_data.append(sample)
                    
                time.sleep(0.001)
                
        # Start recording in separate thread
        record_thread = threading.Thread(target=record_data)
        record_thread.start()
        
    def _save_current_trial(self):
        """Save current trial as individual CSV file"""
        if not self.current_trial_data:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        trial_filename = f"intent_trial_{self.participant_id}_{self.trial_count + 1}_{self.current_action}_{timestamp}.csv"
        trial_path = os.path.join(self.data_dir, trial_filename)
        
        with open(trial_path, 'w', newline='') as f:
            fieldnames = ['timestamp', 'phase_time', 'phase', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
   
            writer.writeheader()
            writer.writerows(self.current_trial_data)
            
        self.status_var.set(f"Trial saved: {trial_filename}")
        
    def quit_application(self):
        if self.is_recording:
            self.acquisition.stop_acquisition()
        self.root.quit()
        
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = IntentAcquisitionGUI()
    app.run()
