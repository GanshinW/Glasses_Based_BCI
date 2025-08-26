#!/usr/bin/env python3
"""
Emotion Recognition Data Acquisition Script
Manual control - each trial saved as individual CSV
Shared JSON for session environment info
"""

import json
import csv
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import numpy as np
import pygame
import os
from signal_acquisition import SignalAcquisition


class EmotionAcquisitionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion Recognition Data Acquisition")
        self.root.geometry("800x700")
        
        # Create data directories
        self.data_dir = os.path.join("data", "emotion")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize pygame mixer for audio
        pygame.mixer.init()
        
        # Data acquisition
        self.acquisition = SignalAcquisition()
        self.is_recording = False
        self.trial_count = 0
        self.participant_id = "P001"
        
        # Music files
        self.music_files = []
        self.current_music_file = ""
        
        # Trial parameters
        self.music_duration = 30  # seconds
        self.recording_start = 20  # start recording at 20s
        self.recording_duration = 10  # record for 10s (20-30s)
        
        # Current trial state
        self.trial_start_time = 0
        self.current_phase = "idle"
        self.current_trial_data = []
        self.waiting_for_rating = False
        
        # Setup GUI
        self._setup_gui()
        self._load_session_info()
        
    def _setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Emotion Recognition Data Acquisition", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Participant info
        ttk.Label(main_frame, text="Participant ID:").grid(row=1, column=0, sticky=tk.W)
        self.participant_entry = ttk.Entry(main_frame)
        self.participant_entry.insert(0, self.participant_id)
        self.participant_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))
        
        # Music files section
        music_frame = ttk.LabelFrame(main_frame, text="Music Files", padding="10")
        music_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        self.music_listbox = tk.Listbox(music_frame, height=6)
        self.music_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        music_scroll = ttk.#!/usr/bin/env python3
"""
Emotion Recognition Data Acquisition Script
Manual control - operator clicks Next Trial button for each trial
Music-induced emotion with 9-point rating scale
Data saved to data/ folder with consistent format
"""

import json
import csv
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import numpy as np
import pygame
import os
from signal_acquisition import SignalAcquisition


class EmotionAcquisitionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion Recognition Data Acquisition")
        self.root.geometry("800x700")
        
        # Create data directory
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize pygame mixer for audio
        pygame.mixer.init()
        
        # Data acquisition
        self.acquisition = SignalAcquisition()
        self.data_buffer = []
        self.is_recording = False
        self.participant_id = "P001"
        
        # Music files
        self.music_files = []
        self.current_music_file = ""
        
        # Trial parameters
        self.music_duration = 30  # seconds
        self.recording_start = 20  # start recording at 20s
        self.recording_duration = 10  # record for 10s (20-30s)
        
        # Current trial state
        self.trial_start_time = 0
        self.current_phase = "idle"
        self.recording_data = []
        self.waiting_for_rating = False
        
        # Setup GUI
        self._setup_gui()
        self._load_existing_data()
        
    def _setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Emotion Recognition Data Acquisition", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Participant info
        ttk.Label(main_frame, text="Participant ID:").grid(row=1, column=0, sticky=tk.W)
        self.participant_entry = ttk.Entry(main_frame)
        self.participant_entry.insert(0, self.participant_id)
        self.participant_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))
        
        # Music files section
        music_frame = ttk.LabelFrame(main_frame, text="Music Files", padding="10")
        music_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        self.music_listbox = tk.Listbox(music_frame, height=6)
        self.music_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        music_scroll = ttk.Scrollbar(music_frame, orient=tk.VERTICAL, command=self.music_listbox.yview)
        music_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.music_listbox.configure(yscrollcommand=music_scroll.set)
        
        music_button_frame = ttk.Frame(music_frame)
        music_button_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        ttk.Button(music_button_frame, text="Add Files", 
                  command=self.add_music_files).pack(fill=tk.X, pady=2)
        ttk.Button(music_button_frame, text="Remove", 
                  command=self.remove_music_file).pack(fill=tk.X, pady=2)
        ttk.Button(music_button_frame, text="Clear All", 
                  command=self.clear_music_files).pack(fill=tk.X, pady=2)
        
        # Trial count info
        ttk.Label(main_frame, text="Completed Trials:").grid(row=3, column=0, sticky=tk.W)
        self.completed_var = tk.StringVar(value="0")
        ttk.Label(main_frame, textvariable=self.completed_var).grid(row=3, column=1, sticky=tk.W)
        
        # Current status display
        self.status_frame = ttk.LabelFrame(main_frame, text="Current Status", padding="20")
        self.status_frame.grid(row=4, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(self.status_frame, text="Add music files and click 'Next Trial'", 
                                     font=("Arial", 20, "bold"))
        self.status_label.pack()
        
        self.phase_label = ttk.Label(self.status_frame, text="", 
                                    font=("Arial", 14))
        self.phase_label.pack()
        
        self.timer_label = ttk.Label(self.status_frame, text="", 
                                    font=("Arial", 12))
        self.timer_label.pack()
        
        # Emotion rating section
        self.rating_frame = ttk.LabelFrame(main_frame, text="Emotion Rating (1-9)", padding="15")
        self.rating_frame.grid(row=5, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))
        
        # Rating scale
        rating_scale_frame = ttk.Frame(self.rating_frame)
        rating_scale_frame.pack()
        
        ttk.Label(rating_scale_frame, text="Negative").pack(side=tk.LEFT)
        
        self.rating_var = tk.IntVar(value=5)
        self.rating_scale = tk.Scale(rating_scale_frame, from_=1, to=9, 
                                    orient=tk.HORIZONTAL, variable=self.rating_var,
                                    length=300)
        self.rating_scale.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(rating_scale_frame, text="Positive").pack(side=tk.LEFT)
        
        # Rating labels
        rating_label_frame = ttk.Frame(self.rating_frame)
        rating_label_frame.pack(pady=10)
        
        ttk.Label(rating_label_frame, text="1-3: Negative", 
                 foreground="red").pack(side=tk.LEFT, padx=20)
        ttk.Label(rating_label_frame, text="4-6: Neutral", 
                 foreground="gray").pack(side=tk.LEFT, padx=20)
        ttk.Label(rating_label_frame, text="7-9: Positive", 
                 foreground="green").pack(side=tk.LEFT, padx=20)
        
        self.submit_rating_button = ttk.Button(self.rating_frame, text="Submit Rating", 
                                              command=self.submit_rating, state=tk.DISABLED)
        self.submit_rating_button.pack(pady=10)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=20)
        
        self.next_trial_button = ttk.Button(button_frame, text="Next Trial", 
                                           command=self.start_next_trial)
        self.next_trial_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(button_frame, text="Save Data", 
                                     command=self.save_data)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        self.quit_button = ttk.Button(button_frame, text="Quit", 
                                     command=self.quit_application)
        self.quit_button.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_bar_var = tk.StringVar(value="Ready - Add music files and click 'Next Trial'")
        status_bar = ttk.Label(main_frame, textvariable=self.status_bar_var, relief=tk.SUNKEN)
        status_bar.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def _load_existing_data(self):
        """Load existing data to continue from where left off"""
        self.participant_id = self.participant_entry.get().strip()
        json_file = os.path.join(self.data_dir, f"emotion_data_{self.participant_id}.json")
        
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    self.data_buffer = data.get('trials', [])
                    self.completed_var.set(str(len(self.data_buffer)))
                    self.status_bar_var.set(f"Loaded {len(self.data_buffer)} existing trials")
            except Exception as e:
                print(f"Error loading existing data: {e}")
        
    def add_music_files(self):
        files = filedialog.askopenfilenames(
            title="Select Music Files",
            filetypes=[("Audio files", "*.mp3 *.wav *.ogg *.m4a"), ("All files", "*.*")]
        )
        
        for file in files:
            if file not in self.music_files:
                self.music_files.append(file)
                self.music_listbox.insert(tk.END, file.split('/')[-1])
                
    def remove_music_file(self):
        selection = self.music_listbox.curselection()
        if selection:
            index = selection[0]
            self.music_files.pop(index)
            self.music_listbox.delete(index)
            
    def clear_music_files(self):
        self.music_files.clear()
        self.music_listbox.delete(0, tk.END)
        
    def start_next_trial(self):
        self.participant_id = self.participant_entry.get().strip()
        if not self.participant_id:
            messagebox.showerror("Error", "Please enter participant ID")
            return
            
        if not self.music_files:
            messagebox.showerror("Error", "Please add music files")
            return
            
        # Select random music file
        self.current_music_file = np.random.choice(self.music_files)
        self.trial_start_time = time.time()
        self.current_phase = "music"
        self.recording_data = []
        self.waiting_for_rating = False
        
        # Disable button during trial
        self.next_trial_button.config(state=tk.DISABLED)
        self.submit_rating_button.config(state=tk.DISABLED)
        
        # Start acquisition if not already started
        if not self.is_recording:
            self.acquisition.start_acquisition()
            self.is_recording = True
            
        # Update display
        self.status_label.config(text=f"Playing: {self.current_music_file.split('/')[-1]}")
        self.phase_label.config(text="Listen to the music")
        self.status_bar_var.set("Music playing...")
        
        # Start music
        try:
            pygame.mixer.music.load(self.current_music_file)
            pygame.mixer.music.play()
        except pygame.error as e:
            messagebox.showerror("Error", f"Cannot play music file: {e}")
            self._reset_trial()
            return
            
        # Schedule recording start
        self.root.after(int(self.recording_start * 1000), self._start_recording)
        
        # Schedule music end
        self.root.after(int(self.music_duration * 1000), self._end_music)
        
        # Update timer
        self._update_timer()
        
    def _start_recording(self):
        if self.current_phase != "music":
            return
            
        self.current_phase = "recording"
        self.phase_label.config(text="Recording EEG - Continue listening")
        
        # Start recording thread
        def record_data():
            start_time = time.time()
            while (time.time() - start_time) < self.recording_duration and self.current_phase == "recording":
                data1, data2 = self.acquisition.read_data()
                if data1 is not None:
                    current_time = time.time()
                    recording_time = current_time - start_time
                    
                    sample = {
                        'timestamp': current_time,
                        'recording_time': recording_time,
                        'channels': data1[:6]
                    }
                    self.recording_data.append(sample)
                    
                time.sleep(0.001)
                
        record_thread = threading.Thread(target=record_data)
        record_thread.start()
        
    def _end_music(self):
        if self.current_phase not in ["music", "recording"]:
            return
            
        pygame.mixer.music.stop()
        self.current_phase = "rating"
        self.waiting_for_rating = True
        
        self.status_label.config(text="Rate your emotional response")
        self.phase_label.config(text="How did the music make you feel?")
        self.timer_label.config(text="")
        
        self.rating_var.set(5)  # Reset to neutral
        self.submit_rating_button.config(state=tk.NORMAL)
        
    def submit_rating(self):
        if not self.waiting_for_rating:
            return
            
        rating = self.rating_var.get()
        
        # Determine emotion category
        if rating <= 3:
            emotion_category = "negative"
        elif rating <= 6:
            emotion_category = "neutral"
        else:
            emotion_category = "positive"
            
        # Store trial data
        trial_data = {
            'trial_id': len(self.data_buffer) + 1,
            'music_file': self.current_music_file.split('/')[-1],
            'music_file_path': self.current_music_file,
            'participant_id': self.participant_id,
            'rating': rating,
            'emotion_category': emotion_category,
            'trial_start_time': self.trial_start_time,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'samples': self.recording_data
        }
        
        self.data_buffer.append(trial_data)
        
        # Update completed count
        self.completed_var.set(str(len(self.data_buffer)))
        
        self._reset_trial()
        
    def _reset_trial(self):
        """Reset trial state for next trial"""
        self.waiting_for_rating = False
        self.current_phase = "idle"
        
        self.next_trial_button.config(state=tk.NORMAL)
        self.submit_rating_button.config(state=tk.DISABLED)
        
        self.status_label.config(text="Trial completed")
        self.phase_label.config(text="Click 'Next Trial' for next trial")
        self.timer_label.config(text="")
        self.status_bar_var.set("Ready for next trial")
        
    def _update_timer(self):
        if self.current_phase not in ["music", "recording"]:
            return
            
        elapsed = time.time() - self.trial_start_time
        
        if self.current_phase == "music":
            if elapsed < self.recording_start:
                remaining = self.recording_start - elapsed
                self.timer_label.config(text=f"Recording starts in: {remaining:.1f}s")
            else:
                remaining = self.music_duration - elapsed
                self.timer_label.config(text=f"Music ends in: {remaining:.1f}s")
        else:  # recording
            remaining = self.music_duration - elapsed
            self.timer_label.config(text=f"Recording... {remaining:.1f}s")
            
        if elapsed < self.music_duration:
            self.root.after(100, self._update_timer)
            
    def save_data(self):
        self.participant_id = self.participant_entry.get().strip()
        if not self.participant_id:
            messagebox.showerror("Error", "Please enter participant ID")
            return
            
        if not self.data_buffer:
            messagebox.showwarning("Warning", "No data to save")
            return
            
        # Save as single JSON file that gets updated
        json_filename = os.path.join(self.data_dir, f"emotion_data_{self.participant_id}.json")
        
        session_info = {
            'participant_id': self.participant_id,
            'task_type': 'emotion_recognition',
            'last_updated': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'sampling_rate': 250,
            'electrode_positions': ['F7', 'FT7', 'T7', 'F8', 'FT8', 'T8'],
            'protocol': {
                'music_duration': self.music_duration,
                'recording_start': self.recording_start,
                'recording_duration': self.recording_duration
            },
            'total_trials': len(self.data_buffer)
        }
        
        data_to_save = {
            'session_info': session_info,
            'trials': self.data_buffer
        }
        
        with open(json_filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)
            
        self.status_bar_var.set(f"Data saved: {json_filename} ({len(self.data_buffer)} trials)")
        messagebox.showinfo("Success", f"Data saved successfully:\n{json_filename}\nTotal trials: {len(self.data_buffer)}")
        
    def quit_application(self):
        if self.is_recording:
            self.acquisition.stop_acquisition()
        pygame.mixer.quit()
        self.root.quit()
        
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = EmotionAcquisitionGUI()
    app.run()
#!/usr/bin/env python3
"""
Emotion Recognition Data Acquisition Script
Music-induced emotion with 9-point rating scale
Recording window: last 10 seconds of 30-second music clips
"""

import json
import csv
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import numpy as np
import pygame
from signal_acquisition import SignalAcquisition


class EmotionAcquisitionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion Recognition Data Acquisition")
        self.root.geometry("800x700")
        
        # Initialize pygame mixer for audio
        pygame.mixer.init()
        
        # Data acquisition
        self.acquisition = SignalAcquisition()
        self.data_buffer = []
        self.is_recording = False
        self.current_trial = 0
        self.participant_id = "P001"
        
        # Music files
        self.music_files = []
        self.current_music_file = ""
        
        # Trial parameters
        self.music_duration = 30  # seconds
        self.recording_start = 20  # start recording at 20s
        self.recording_duration = 10  # record for 10s (20-30s)
        self.rest_duration = 15  # rest between trials
        
        # Current trial state
        self.trial_start_time = 0
        self.current_phase = "idle"
        self.recording_data = []
        
        # Setup GUI
        self._setup_gui()
        
    def _setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Emotion Recognition Data Acquisition", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Participant info
        ttk.Label(main_frame, text="Participant ID:").grid(row=1, column=0, sticky=tk.W)
        self.participant_entry = ttk.Entry(main_frame)
        self.participant_entry.insert(0, self.participant_id)
        self.participant_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))
        
        # Music files section
        music_frame = ttk.LabelFrame(main_frame, text="Music Files", padding="10")
        music_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        self.music_listbox = tk.Listbox(music_frame, height=6)
        self.music_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        music_scroll = ttk.Scrollbar(music_frame, orient=tk.VERTICAL, command=self.music_listbox.yview)
        music_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.music_listbox.configure(yscrollcommand=music_scroll.set)
        
        music_button_frame = ttk.Frame(music_frame)
        music_button_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        ttk.Button(music_button_frame, text="Add Files", 
                  command=self.add_music_files).pack(fill=tk.X, pady=2)
        ttk.Button(music_button_frame, text="Remove", 
                  command=self.remove_music_file).pack(fill=tk.X, pady=2)
        ttk.Button(music_button_frame, text="Clear All", 
                  command=self.clear_music_files).pack(fill=tk.X, pady=2)
        
        # Trial progress
        ttk.Label(main_frame, text="Trial Progress:").grid(row=3, column=0, sticky=tk.W)
        self.progress_var = tk.StringVar(value="0/0")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(row=3, column=1, sticky=tk.W)
        
        # Current status display
        self.status_frame = ttk.LabelFrame(main_frame, text="Current Status", padding="20")
        self.status_frame.grid(row=4, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(self.status_frame, text="Ready to start", 
                                     font=("Arial", 20, "bold"))
        self.status_label.pack()
        
        self.phase_label = ttk.Label(self.status_frame, text="", 
                                    font=("Arial", 14))
        self.phase_label.pack()
        
        self.timer_label = ttk.Label(self.status_frame, text="", 
                                    font=("Arial", 12))
        self.timer_label.pack()
        
        # Emotion rating section
        self.rating_frame = ttk.LabelFrame(main_frame, text="Emotion Rating (1-9)", padding="15")
        self.rating_frame.grid(row=5, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))
        
        # Rating scale
        rating_scale_frame = ttk.Frame(self.rating_frame)
        rating_scale_frame.pack()
        
        ttk.Label(rating_scale_frame, text="Negative").pack(side=tk.LEFT)
        
        self.rating_var = tk.IntVar(value=5)
        self.rating_scale = tk.Scale(rating_scale_frame, from_=1, to=9, 
                                    orient=tk.HORIZONTAL, variable=self.rating_var,
                                    length=300)
        self.rating_scale.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(rating_scale_frame, text="Positive").pack(side=tk.LEFT)
        
        # Rating labels
        rating_label_frame = ttk.Frame(self.rating_frame)
        rating_label_frame.pack(pady=10)
        
        ttk.Label(rating_label_frame, text="1-3: Negative", 
                 foreground="red").pack(side=tk.LEFT, padx=20)
        ttk.Label(rating_label_frame, text="4-6: Neutral", 
                 foreground="gray").pack(side=tk.LEFT, padx=20)
        ttk.Label(rating_label_frame, text="7-9: Positive", 
                 foreground="green").pack(side=tk.LEFT, padx=20)
        
        self.submit_rating_button = ttk.Button(self.rating_frame, text="Submit Rating", 
                                              command=self.submit_rating, state=tk.DISABLED)
        self.submit_rating_button.pack(pady=10)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=20)
        
        self.start_button = ttk.Button(button_frame, text="Start Experiment", 
                                      command=self.start_experiment)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Experiment", 
                                     command=self.stop_experiment, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(button_frame, text="Save Data", 
                                     command=self.save_data, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_bar_var = tk.StringVar(value="Ready - Add music files to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_bar_var, relief=tk.SUNKEN)
        status_bar.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def add_music_files(self):
        files = filedialog.askopenfilenames(
            title="Select Music Files",
            filetypes=[("Audio files", "*.mp3 *.wav *.ogg *.m4a"), ("All files", "*.*")]
        )
        
        for file in files:
            if file not in self.music_files:
                self.music_files.append(file)
                self.music_listbox.insert(tk.END, file.split('/')[-1])
                
        self.progress_var.set(f"0/{len(self.music_files)}")
        
    def remove_music_file(self):
        selection = self.music_listbox.curselection()
        if selection:
            index = selection[0]
            self.music_files.pop(index)
            self.music_listbox.delete(index)
            self.progress_var.set(f"0/{len(self.music_files)}")
            
    def clear_music_files(self):
        self.music_files.clear()
        self.music_listbox.delete(0, tk.END)
        self.progress_var.set("0/0")
        
    def start_experiment(self):
        self.participant_id = self.participant_entry.get().strip()
        if not self.participant_id:
            messagebox.showerror("Error", "Please enter participant ID")
            return
            
        if not self.music_files:
            messagebox.showerror("Error", "Please add music files")
            return
            
        # Shuffle music files
        np.random.shuffle(self.music_files)
        
        self.is_recording = True
        self.current_trial = 0
        self.data_buffer = []
        
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        
        self.acquisition.start_acquisition()
        self.status_bar_var.set("Experiment running")
        
        self._run_next_trial()
        
    def stop_experiment(self):
        self.is_recording = False
        pygame.mixer.music.stop()
        self.acquisition.stop_acquisition()
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL)
        self.submit_rating_button.config(state=tk.DISABLED)
        
        self.status_bar_var.set("Experiment stopped")
        self.status_label.config(text="Experiment stopped")
        self.phase_label.config(text="")
        self.timer_label.config(text="")
        
    def _run_next_trial(self):
        if not self.is_recording or self.current_trial >= len(self.music_files):
            self.stop_experiment()
            return
            
        self.current_music_file = self.music_files[self.current_trial]
        self.trial_start_time = time.time()
        self.current_phase = "music"
        self.recording_data = []
        
        self.progress_var.set(f"{self.current_trial + 1}/{len(self.music_files)}")
        
        # Update display
        self.status_label.config(text=f"Playing: {self.current_music_file.split('/')[-1]}")
        self.phase_label.config(text="Listen to the music")
        self.submit_rating_button.config(state=tk.DISABLED)
        
        # Start music
        try:
            pygame.mixer.music.load(self.current_music_file)
            pygame.mixer.music.play()
        except pygame.error as e:
            messagebox.showerror("Error", f"Cannot play music file: {e}")
            self._skip_trial()
            return
            
        # Schedule recording start
        self.root.after(int(self.recording_start * 1000), self._start_recording)
        
        # Schedule music end
        self.root.after(int(self.music_duration * 1000), self._end_music)
        
        # Update timer
        self._update_timer()
        
    def _start_recording(self):
        if not self.is_recording:
            return
            
        self.current_phase = "recording"
        self.phase_label.config(text="Recording EEG - Continue listening")
        
        # Start recording thread
        def record_data():
            start_time = time.time()
            while (time.time() - start_time) < self.recording_duration and self.is_recording:
                data1, data2 = self.acquisition.read_data()
                if data1 is not None:
                    current_time = time.time()
                    recording_time = current_time - start_time
                    
                    sample = {
                        'timestamp': current_time,
                        'recording_time': recording_time,
                        'channels': data1[:6]
                    }
                    self.recording_data.append(sample)
                    
                time.sleep(0.001)
                
        record_thread = threading.Thread(target=record_data)
        record_thread.start()
        
    def _end_music(self):
        if not self.is_recording:
            return
            
        pygame.mixer.music.stop()
        self.current_phase = "rating"
        
        self.status_label.config(text="Rate your emotional response")
        self.phase_label.config(text="How did the music make you feel?")
        self.timer_label.config(text="")
        
        self.rating_var.set(5)  # Reset to neutral
        self.submit_rating_button.config(state=tk.NORMAL)
        
    def submit_rating(self):
        if not self.is_recording:
            return
            
        rating = self.rating_var.get()
        
        # Determine emotion category
        if rating <= 3:
            emotion_category = "negative"
        elif rating <= 6:
            emotion_category = "neutral"
        else:
            emotion_category = "positive"
            
        # Store trial data
        trial_data = {
            'trial_id': self.current_trial + 1,
            'music_file': self.current_music_file,
            'participant_id': self.participant_id,
            'rating': rating,
            'emotion_category': emotion_category,
            'trial_start_time': self.trial_start_time,
            'samples': self.recording_data
        }
        
        self.data_buffer.append(trial_data)
        
        self.submit_rating_button.config(state=tk.DISABLED)
        self.current_trial += 1
        
        # Start rest period
        self._start_rest()
        
    def _start_rest(self):
        if not self.is_recording:
            return
            
        self.current_phase = "rest"
        self.status_label.config(text="Rest Period")
        self.phase_label.config(text="Take a break - relax")
        
        # Rest timer countdown
        def rest_countdown(remaining):
            if not self.is_recording:
                return
                
            self.timer_label.config(text=f"Next trial in: {remaining}s")
            
            if remaining > 0:
                self.root.after(1000, lambda: rest_countdown(remaining - 1))
            else:
                self._run_next_trial()
                
        rest_countdown(self.rest_duration)
        
    def _skip_trial(self):
        self.current_trial += 1
        self._run_next_trial()
        
    def _update_timer(self):
        if not self.is_recording or self.current_phase not in ["music", "recording"]:
            return
            
        elapsed = time.time() - self.trial_start_time
        
        if self.current_phase == "music":
            if elapsed < self.recording_start:
                remaining = self.recording_start - elapsed
                self.timer_label.config(text=f"Recording starts in: {remaining:.1f}s")
            else:
                remaining = self.music_duration - elapsed
                self.timer_label.config(text=f"Music ends in: {remaining:.1f}s")
        else:  # recording
            remaining = self.music_duration - elapsed
            self.timer_label.config(text=f"Recording... {remaining:.1f}s")
            
        if elapsed < self.music_duration:
            self.root.after(100, self._update_timer)
            
    def save_data(self):
        if not self.data_buffer:
            messagebox.showwarning("Warning", "No data to save")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_filename = f"emotion_data_{self.participant_id}_{timestamp}.json"
        
        session_info = {
            'participant_id': self.participant_id,
            'timestamp': timestamp,
            'sampling_rate': 250,
            'electrode_positions': ['F7', 'FT7', 'T7', 'F8', 'FT8', 'T8'],
            'protocol': {
                'music_duration': self.music_duration,
                'recording_start': self.recording_start,
                'recording_duration': self.recording_duration,
                'rest_duration': self.rest_duration
            }
        }
        
        data_to_save = {
            'session_info': session_info,
            'trials': self.data_buffer
        }
        
        with open(json_filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)
            
        # Save as CSV#!/usr/bin/env python3
"""
Emotion Recognition Data Acquisition Script
Manual control - each trial saved as individual CSV
Shared JSON for session environment info
"""

import json
import csv
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import numpy as np
import pygame
import os
from signal_acquisition import SignalAcquisition


class EmotionAcquisitionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion Recognition Data Acquisition")
        self.root.geometry("800x700")
        
        # Create data directories
        self.data_dir = os.path.join("data", "emotion")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize pygame mixer for audio
        pygame.mixer.init()
        
        # Data acquisition
        self.acquisition = SignalAcquisition()
        self.is_recording = False
        self.trial_count = 0
        self.participant_id = "P001"
        
        # Music files
        self.music_files = []
        self.current_music_file = ""
        
        # Trial parameters
        self.music_duration = 30  # seconds
        self.recording_start = 20  # start recording at 20s
        self.recording_duration = 10  # record for 10s (20-30s)
        
        # Current trial state
        self.trial_start_time = 0
        self.current_phase = "idle"
        self.current_trial_data = []
        self.waiting_for_rating = False
        
        # Setup GUI
        self._setup_gui()
        self._load_session_info()
        
    def _setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Emotion Recognition Data Acquisition", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Participant info
        ttk.Label(main_frame, text="Participant ID:").grid(row=1, column=0, sticky=tk.W)
        self.participant_entry = ttk.Entry(main_frame)
        self.participant_entry.insert(0, self.participant_id)
        self.participant_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))
        
        # Music files section
        music_frame = ttk.LabelFrame(main_frame, text="Music Files", padding="10")
        music_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        self.music_listbox = tk.Listbox(music_frame, height=6)
        self.music_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        music_scroll = ttk.Scrollbar(music_frame, orient=tk.VERTICAL, command=self.music_listbox.yview)
        music_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.music_listbox.configure(yscrollcommand=music_scroll.set)
        
        music_button_frame = ttk.Frame(music_frame)
        music_button_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        ttk.Button(music_button_frame, text="Add Files", 
                  command=self.add_music_files).pack(fill=tk.X, pady=2)
        ttk.Button(music_button_frame, text="Remove", 
                  command=self.remove_music_file).pack(fill=tk.X, pady=2)
        ttk.Button(music_button_frame, text="Clear All", 
                  command=self.clear_music_files).pack(fill=tk.X, pady=2)
        
        # Trial count info
        ttk.Label(main_frame, text="Completed Trials:").grid(row=3, column=0, sticky=tk.W)
        self.trial_count_var = tk.StringVar(value="0")
        ttk.Label(main_frame, textvariable=self.trial_count_var).grid(row=3, column=1, sticky=tk.W)
        
        # Current status display
        self.status_frame = ttk.LabelFrame(main_frame, text="Current Status", padding="20")
        self.status_frame.grid(row=4, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(self.status_frame, text="Add music files and click 'Next Trial'", 
                                     font=("Arial", 20, "bold"))
        self.status_label.pack()
        
        self.phase_label = ttk.Label(self.status_frame, text="", 
                                    font=("Arial", 14))
        self.phase_label.pack()
        
        self.timer_label = ttk.Label(self.status_frame, text="", 
                                    font=("Arial", 12))
        self.timer_label.pack()
        
        # Emotion rating section
        self.rating_frame = ttk.LabelFrame(main_frame, text="Emotion Rating (1-9)", padding="15")
        self.rating_frame.grid(row=5, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))
        
        # Rating scale
        rating_scale_frame = ttk.Frame(self.rating_frame)
        rating_scale_frame.pack()
        
        ttk.Label(rating_scale_frame, text="Negative").pack(side=tk.LEFT)
        
        self.rating_var = tk.IntVar(value=5)
        self.rating_scale = tk.Scale(rating_scale_frame, from_=1, to=9, 
                                    orient=tk.HORIZONTAL, variable=self.rating_var,
                                    length=300)
        self.rating_scale.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(rating_scale_frame, text="Positive").pack(side=tk.LEFT)
        
        # Rating labels
        rating_label_frame = ttk.Frame(self.rating_frame)
        rating_label_frame.pack(pady=10)
        
        ttk.Label(rating_label_frame, text="1-3: Negative", 
                 foreground="red").pack(side=tk.LEFT, padx=20)
        ttk.Label(rating_label_frame, text="4-6: Neutral", 
                 foreground="gray").pack(side=tk.LEFT, padx=20)
        ttk.Label(rating_label_frame, text="7-9: Positive", 
                 foreground="green").pack(side=tk.LEFT, padx=20)
        
        self.submit_rating_button = ttk.Button(self.rating_frame, text="Submit Rating", 
                                              command=self.submit_rating, state=tk.DISABLED)
        self.submit_rating_button.pack(pady=10)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=20)
        
        self.next_trial_button = ttk.Button(button_frame, text="Next Trial", 
                                           command=self.start_next_trial)
        self.next_trial_button.pack(side=tk.LEFT, padx=5)
        
        self.quit_button = ttk.Button(button_frame, text="Quit", 
                                     command=self.quit_application)
        self.quit_button.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_bar_var = tk.StringVar(value="Ready - Add music files and click 'Next Trial'")
        status_bar = ttk.Label(main_frame, textvariable=self.status_bar_var, relief=tk.SUNKEN)
        status_bar.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
        
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
            'task_type': 'emotion_recognition',
            'participant_id': self.participant_id,
            'last_updated': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'sampling_rate': 250,
            'electrode_positions': ['F7', 'FT7', 'T7', 'F8', 'FT8', 'T8'],
            'protocol': {
                'music_duration': self.music_duration,
                'recording_start': self.recording_start,
                'recording_duration': self.recording_duration
            },
            'total_trials': self.trial_count
        }
        
        with open(self.session_file, 'w') as f:
            json.dump(session_info, f, indent=2)
        
    def add_music_files(self):
        files = filedialog.askopenfilenames(
            title="Select Music Files",
            filetypes=[("Audio files", "*.mp3 *.wav *.ogg *.m4a"), ("All files", "*.*")]
        )
        
        for file in files:
            if file not in self.music_files:
                self.music_files.append(file)
                self.music_listbox.insert(tk.END, file.split('/')[-1])
                
    def remove_music_file(self):
        selection = self.music_listbox.curselection()
        if selection:
            index = selection[0]
            self.music_files.pop(index)
            self.music_listbox.delete(index)
            
    def clear_music_files(self):
        self.music_files.clear()
        self.music_listbox.delete(0, tk.END)
        
    def start_next_trial(self):
        self.participant_id = self.participant_entry.get().strip()
        if not self.participant_id:
            messagebox.showerror("Error", "Please enter participant ID")
            return
            
        if not self.music_files:
            messagebox.showerror("Error", "Please add music files")
            return
            
        # Select random music file
        self.current_music_file = np.random.choice(self.music_files)
        self.trial_start_time = time.time()
        self.current_phase = "music"
        self.current_trial_data = []
        self.waiting_for_rating = False
        
        # Disable button during trial
        self.next_trial_button.config(state=tk.DISABLED)
        self.submit_rating_button.config(state=tk.DISABLED)
        
        # Start acquisition if not already started
        if not self.is_recording:
            self.acquisition.start_acquisition()
            self.is_recording = True
            
        # Update display
        self.status_label.config(text=f"Playing: {self.current_music_file.split('/')[-1]}")
        self.phase_label.config(text="Listen to the music")
        self.status_bar_var.set("Music playing...")
        
        # Start music
        try:
            pygame.mixer.music.load(self.current_music_file)
            pygame.mixer.music.play()
        except pygame.error as e:
            messagebox.showerror("Error", f"Cannot play music file: {e}")
            self._reset_trial()
            return
            
        # Schedule recording start
        self.root.after(int(self.recording_start * 1000), self._start_recording)
        
        # Schedule music end
        self.root.after(int(self.music_duration * 1000), self._end_music)
        
        # Update timer
        self._update_timer()
        
    def _start_recording(self):
        if self.current_phase != "music":
            return
            
        self.current_phase = "recording"
        self.phase_label.config(text="Recording EEG - Continue listening")
        
        # Start recording thread
        def record_data():
            start_time = time.time()
            while (time.time() - start_time) < self.recording_duration and self.current_phase == "recording":
                data1, data2 = self.acquisition.read_data()
                if data1 is not None:
                    current_time = time.time()
                    recording_time = current_time - start_time
                    
                    sample = {
                        'timestamp': current_time,
                        'recording_time': recording_time,
                        'ch1': data1[0], 'ch2': data1[1], 'ch3': data1[2],
                        'ch4': data1[3], 'ch5': data1[4], 'ch6': data1[5]
                    }
                    self.current_trial_data.append(sample)
                    
                time.sleep(0.001)
                
        record_thread = threading.Thread(target=record_data)
        record_thread.start()
        
    def _end_music(self):
        if self.current_phase not in ["music", "recording"]:
            return
            
        pygame.mixer.music.stop()
        self.current_phase = "rating"
        self.waiting_for_rating = True
        
        self.status_label.config(text="Rate your emotional response")
        self.phase_label.config(text="How did the music make you feel?")
        self.timer_label.config(text="")
        
        self.rating_var.set(5)  # Reset to neutral
        self.submit_rating_button.config(state=tk.NORMAL)
        
    def submit_rating(self):
        if not self.waiting_for_rating:
            return
            
        rating = self.rating_var.get()
        
        # Determine emotion category
        if rating <= 3:
            emotion_category = "negative"
        elif rating <= 6:
            emotion_category = "neutral"
        else:
            emotion_category = "positive"
            
        # Save this trial
        self._save_current_trial(rating, emotion_category)
        
        # Update counters
        self.trial_count += 1
        self.trial_count_var.set(str(self.trial_count))
        self._update_session_info()
        
        self._reset_trial()
        
    def _save_current_trial(self, rating, emotion_category):
        """Save current trial as individual CSV file"""
        if not self.current_trial_data:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        music_name = os.path.splitext(os.path.basename(self.current_music_file))[0]
        trial_filename = f"emotion_trial_{self.participant_id}_{self.trial_count + 1}_{emotion_category}_{rating}_{music_name}_{timestamp}.csv"
        trial_path = os.path.join(self.data_dir, trial_filename)
        
        with open(trial_path, 'w', newline='') as f:
            fieldnames = ['timestamp', 'recording_time', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
         
            writer.writeheader()
            writer.writerows(self.current_trial_data)
            
        self.status_bar_var.set(f"Trial saved: {trial_filename}")
        
    def _reset_trial(self):
        """Reset trial state for next trial"""
        self.waiting_for_rating = False
        self.current_phase = "idle"
        
        self.next_trial_button.config(state=tk.NORMAL)
        self.submit_rating_button.config(state=tk.DISABLED)
        
        self.status_label.config(text="Trial completed")
        self.phase_label.config(text="Click 'Next Trial' for next trial")
        self.timer_label.config(text="")
        self.status_bar_var.set("Ready for next trial")
        
    def _update_timer(self):
        if self.current_phase not in ["music", "recording"]:
            return
            
        elapsed = time.time() - self.trial_start_time
        
        if self.current_phase == "music":
            if elapsed < self.recording_start:
                remaining = self.recording_start - elapsed
                self.timer_label.config(text=f"Recording starts in: {remaining:.1f}s")
            else:
                remaining = self.music_duration - elapsed
                self.timer_label.config(text=f"Music ends in: {remaining:.1f}s")
        else:  # recording
            remaining = self.music_duration - elapsed
            self.timer_label.config(text=f"Recording... {remaining:.1f}s")
            
        if elapsed < self.music_duration:
            self.root.after(100, self._update_timer)
            
    def quit_application(self):
        if self.is_recording:
            self.acquisition.stop_acquisition()
        pygame.mixer.quit()
        self.root.quit()
        
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = EmotionAcquisitionGUI()
    app.run()
#!/usr/bin/env python3
"""
Emotion Recognition Data Acquisition Script
Manual control - operator clicks Next Trial button for each trial
Music-induced emotion with 9-point rating scale
Data saved to data/ folder with consistent format
"""

import json
import csv
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import numpy as np
import pygame
import os
from signal_acquisition import SignalAcquisition


class EmotionAcquisitionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion Recognition Data Acquisition")
        self.root.geometry("800x700")
        
        # Create data directory
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize pygame mixer for audio
        pygame.mixer.init()
        
        # Data acquisition
        self.acquisition = SignalAcquisition()
        self.data_buffer = []
        self.is_recording = False
        self.participant_id = "P001"
        
        # Music files
        self.music_files = []
        self.current_music_file = ""
        
        # Trial parameters
        self.music_duration = 30  # seconds
        self.recording_start = 20  # start recording at 20s
        self.recording_duration = 10  # record for 10s (20-30s)
        
        # Current trial state
        self.trial_start_time = 0
        self.current_phase = "idle"
        self.recording_data = []
        self.waiting_for_rating = False
        
        # Setup GUI
        self._setup_gui()
        self._load_existing_data()
        
    def _setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Emotion Recognition Data Acquisition", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Participant info
        ttk.Label(main_frame, text="Participant ID:").grid(row=1, column=0, sticky=tk.W)
        self.participant_entry = ttk.Entry(main_frame)
        self.participant_entry.insert(0, self.participant_id)
        self.participant_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))
        
        # Music files section
        music_frame = ttk.LabelFrame(main_frame, text="Music Files", padding="10")
        music_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        self.music_listbox = tk.Listbox(music_frame, height=6)
        self.music_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        music_scroll = ttk.Scrollbar(music_frame, orient=tk.VERTICAL, command=self.music_listbox.yview)
        music_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.music_listbox.configure(yscrollcommand=music_scroll.set)
        
        music_button_frame = ttk.Frame(music_frame)
        music_button_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        ttk.Button(music_button_frame, text="Add Files", 
                  command=self.add_music_files).pack(fill=tk.X, pady=2)
        ttk.Button(music_button_frame, text="Remove", 
                  command=self.remove_music_file).pack(fill=tk.X, pady=2)
        ttk.Button(music_button_frame, text="Clear All", 
                  command=self.clear_music_files).pack(fill=tk.X, pady=2)
        
        # Trial count info
        ttk.Label(main_frame, text="Completed Trials:").grid(row=3, column=0, sticky=tk.W)
        self.completed_var = tk.StringVar(value="0")
        ttk.Label(main_frame, textvariable=self.completed_var).grid(row=3, column=1, sticky=tk.W)
        
        # Current status display
        self.status_frame = ttk.LabelFrame(main_frame, text="Current Status", padding="20")
        self.status_frame.grid(row=4, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(self.status_frame, text="Add music files and click 'Next Trial'", 
                                     font=("Arial", 20, "bold"))
        self.status_label.pack()
        
        self.phase_label = ttk.Label(self.status_frame, text="", 
                                    font=("Arial", 14))
        self.phase_label.pack()
        
        self.timer_label = ttk.Label(self.status_frame, text="", 
                                    font=("Arial", 12))
        self.timer_label.pack()
        
        # Emotion rating section
        self.rating_frame = ttk.LabelFrame(main_frame, text="Emotion Rating (1-9)", padding="15")
        self.rating_frame.grid(row=5, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))
        
        # Rating scale
        rating_scale_frame = ttk.Frame(self.rating_frame)
        rating_scale_frame.pack()
        
        ttk.Label(rating_scale_frame, text="Negative").pack(side=tk.LEFT)
        
        self.rating_var = tk.IntVar(value=5)
        self.rating_scale = tk.Scale(rating_scale_frame, from_=1, to=9, 
                                    orient=tk.HORIZONTAL, variable=self.rating_var,
                                    length=300)
        self.rating_scale.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(rating_scale_frame, text="Positive").pack(side=tk.LEFT)
        
        # Rating labels
        rating_label_frame = ttk.Frame(self.rating_frame)
        rating_label_frame.pack(pady=10)
        
        ttk.Label(rating_label_frame, text="1-3: Negative", 
                 foreground="red").pack(side=tk.LEFT, padx=20)
        ttk.Label(rating_label_frame, text="4-6: Neutral", 
                 foreground="gray").pack(side=tk.LEFT, padx=20)
        ttk.Label(rating_label_frame, text="7-9: Positive", 
                 foreground="green").pack(side=tk.LEFT, padx=20)
        
        self.submit_rating_button = ttk.Button(self.rating_frame, text="Submit Rating", 
                                              command=self.submit_rating, state=tk.DISABLED)
        self.submit_rating_button.pack(pady=10)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=20)
        
        self.next_trial_button = ttk.Button(button_frame, text="Next Trial", 
                                           command=self.start_next_trial)
        self.next_trial_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(button_frame, text="Save Data", 
                                     command=self.save_data)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        self.quit_button = ttk.Button(button_frame, text="Quit", 
                                     command=self.quit_application)
        self.quit_button.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_bar_var = tk.StringVar(value="Ready - Add music files and click 'Next Trial'")
        status_bar = ttk.Label(main_frame, textvariable=self.status_bar_var, relief=tk.SUNKEN)
        status_bar.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def _load_existing_data(self):
        """Load existing data to continue from where left off"""
        self.participant_id = self.participant_entry.get().strip()
        json_file = os.path.join(self.data_dir, f"emotion_data_{self.participant_id}.json")
        
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    self.data_buffer = data.get('trials', [])
                    self.completed_var.set(str(len(self.data_buffer)))
                    self.status_bar_var.set(f"Loaded {len(self.data_buffer)} existing trials")
            except Exception as e:
                print(f"Error loading existing data: {e}")
        
    def add_music_files(self):
        files = filedialog.askopenfilenames(
            title="Select Music Files",
            filetypes=[("Audio files", "*.mp3 *.wav *.ogg *.m4a"), ("All files", "*.*")]
        )
        
        for file in files:
            if file not in self.music_files:
                self.music_files.append(file)
                self.music_listbox.insert(tk.END, file.split('/')[-1])
                
    def remove_music_file(self):
        selection = self.music_listbox.curselection()
        if selection:
            index = selection[0]
            self.music_files.pop(index)
            self.music_listbox.delete(index)
            
    def clear_music_files(self):
        self.music_files.clear()
        self.music_listbox.delete(0, tk.END)
        
    def start_next_trial(self):
        self.participant_id = self.participant_entry.get().strip()
        if not self.participant_id:
            messagebox.showerror("Error", "Please enter participant ID")
            return
            
        if not self.music_files:
            messagebox.showerror("Error", "Please add music files")
            return
            
        # Select random music file
        self.current_music_file = np.random.choice(self.music_files)
        self.trial_start_time = time.time()
        self.current_phase = "music"
        self.recording_data = []
        self.waiting_for_rating = False
        
        # Disable button during trial
        self.next_trial_button.config(state=tk.DISABLED)
        self.submit_rating_button.config(state=tk.DISABLED)
        
        # Start acquisition if not already started
        if not self.is_recording:
            self.acquisition.start_acquisition()
            self.is_recording = True
            
        # Update display
        self.status_label.config(text=f"Playing: {self.current_music_file.split('/')[-1]}")
        self.phase_label.config(text="Listen to the music")
        self.status_bar_var.set("Music playing...")
        
        # Start music
        try:
            pygame.mixer.music.load(self.current_music_file)
            pygame.mixer.music.play()
        except pygame.error as e:
            messagebox.showerror("Error", f"Cannot play music file: {e}")
            self._reset_trial()
            return
            
        # Schedule recording start
        self.root.after(int(self.recording_start * 1000), self._start_recording)
        
        # Schedule music end
        self.root.after(int(self.music_duration * 1000), self._end_music)
        
        # Update timer
        self._update_timer()
        
    def _start_recording(self):
        if self.current_phase != "music":
            return
            
        self.current_phase = "recording"
        self.phase_label.config(text="Recording EEG - Continue listening")
        
        # Start recording thread
        def record_data():
            start_time = time.time()
            while (time.time() - start_time) < self.recording_duration and self.current_phase == "recording":
                data1, data2 = self.acquisition.read_data()
                if data1 is not None:
                    current_time = time.time()
                    recording_time = current_time - start_time
                    
                    sample = {
                        'timestamp': current_time,
                        'recording_time': recording_time,
                        'channels': data1[:6]
                    }
                    self.recording_data.append(sample)
                    
                time.sleep(0.001)
                
        record_thread = threading.Thread(target=record_data)
        record_thread.start()
        
    def _end_music(self):
        if self.current_phase not in ["music", "recording"]:
            return
            
        pygame.mixer.music.stop()
        self.current_phase = "rating"
        self.waiting_for_rating = True
        
        self.status_label.config(text="Rate your emotional response")
        self.phase_label.config(text="How did the music make you feel?")
        self.timer_label.config(text="")
        
        self.rating_var.set(5)  # Reset to neutral
        self.submit_rating_button.config(state=tk.NORMAL)
        
    def submit_rating(self):
        if not self.waiting_for_rating:
            return
            
        rating = self.rating_var.get()
        
        # Determine emotion category
        if rating <= 3:
            emotion_category = "negative"
        elif rating <= 6:
            emotion_category = "neutral"
        else:
            emotion_category = "positive"
            
        # Store trial data
        trial_data = {
            'trial_id': len(self.data_buffer) + 1,
            'music_file': self.current_music_file.split('/')[-1],
            'music_file_path': self.current_music_file,
            'participant_id': self.participant_id,
            'rating': rating,
            'emotion_category': emotion_category,
            'trial_start_time': self.trial_start_time,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'samples': self.recording_data
        }
        
        self.data_buffer.append(trial_data)
        
        # Update completed count
        self.completed_var.set(str(len(self.data_buffer)))
        
        self._reset_trial()
        
    def _reset_trial(self):
        """Reset trial state for next trial"""
        self.waiting_for_rating = False
        self.current_phase = "idle"
        
        self.next_trial_button.config(state=tk.NORMAL)
        self.submit_rating_button.config(state=tk.DISABLED)
        
        self.status_label.config(text="Trial completed")
        self.phase_label.config(text="Click 'Next Trial' for next trial")
        self.timer_label.config(text="")
        self.status_bar_var.set("Ready for next trial")
        
    def _update_timer(self):
        if self.current_phase not in ["music", "recording"]:
            return
            
        elapsed = time.time() - self.trial_start_time
        
        if self.current_phase == "music":
            if elapsed < self.recording_start:
                remaining = self.recording_start - elapsed
                self.timer_label.config(text=f"Recording starts in: {remaining:.1f}s")
            else:
                remaining = self.music_duration - elapsed
                self.timer_label.config(text=f"Music ends in: {remaining:.1f}s")
        else:  # recording
            remaining = self.music_duration - elapsed
            self.timer_label.config(text=f"Recording... {remaining:.1f}s")
            
        if elapsed < self.music_duration:
            self.root.after(100, self._update_timer)
            
    def save_data(self):
        self.participant_id = self.participant_entry.get().strip()
        if not self.participant_id:
            messagebox.showerror("Error", "Please enter participant ID")
            return
            
        if not self.data_buffer:
            messagebox.showwarning("Warning", "No data to save")
            return
            
        # Save as single JSON file that gets updated
        json_filename = os.path.join(self.data_dir, f"emotion_data_{self.participant_id}.json")
        
        session_info = {
            'participant_id': self.participant_id,
            'task_type': 'emotion_recognition',
            'last_updated': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'sampling_rate': 250,
            'electrode_positions': ['F7', 'FT7', 'T7', 'F8', 'FT8', 'T8'],
            'protocol': {
                'music_duration': self.music_duration,
                'recording_start': self.recording_start,
                'recording_duration': self.recording_duration
            },
            'total_trials': len(self.data_buffer)
        }
        
        data_to_save = {
            'session_info': session_info,
            'trials': self.data_buffer
        }
        
        with open(json_filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)
            
        self.status_bar_var.set(f"Data saved: {json_filename} ({len(self.data_buffer)} trials)")
        messagebox.showinfo("Success", f"Data saved successfully:\n{json_filename}\nTotal trials: {len(self.data_buffer)}")
        
    def quit_application(self):
        if self.is_recording:
            self.acquisition.stop_acquisition()
        pygame.mixer.quit()
        self.root.quit()
        
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = EmotionAcquisitionGUI()
    app.run()
#!/usr/bin/env python3
"""
Emotion Recognition Data Acquisition Script
Music-induced emotion with 9-point rating scale
Recording window: last 10 seconds of 30-second music clips
"""

import json
import csv
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import numpy as np
import pygame
from signal_acquisition import SignalAcquisition


class EmotionAcquisitionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion Recognition Data Acquisition")
        self.root.geometry("800x700")
        
        # Initialize pygame mixer for audio
        pygame.mixer.init()
        
        # Data acquisition
        self.acquisition = SignalAcquisition()
        self.data_buffer = []
        self.is_recording = False
        self.current_trial = 0
        self.participant_id = "P001"
        
        # Music files
        self.music_files = []
        self.current_music_file = ""
        
        # Trial parameters
        self.music_duration = 30  # seconds
        self.recording_start = 20  # start recording at 20s
        self.recording_duration = 10  # record for 10s (20-30s)
        self.rest_duration = 15  # rest between trials
        
        # Current trial state
        self.trial_start_time = 0
        self.current_phase = "idle"
        self.recording_data = []
        
        # Setup GUI
        self._setup_gui()
        
    def _setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Emotion Recognition Data Acquisition", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Participant info
        ttk.Label(main_frame, text="Participant ID:").grid(row=1, column=0, sticky=tk.W)
        self.participant_entry = ttk.Entry(main_frame)
        self.participant_entry.insert(0, self.participant_id)
        self.participant_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))
        
        # Music files section
        music_frame = ttk.LabelFrame(main_frame, text="Music Files", padding="10")
        music_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        self.music_listbox = tk.Listbox(music_frame, height=6)
        self.music_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        music_scroll = ttk.Scrollbar(music_frame, orient=tk.VERTICAL, command=self.music_listbox.yview)
        music_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.music_listbox.configure(yscrollcommand=music_scroll.set)
        
        music_button_frame = ttk.Frame(music_frame)
        music_button_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        ttk.Button(music_button_frame, text="Add Files", 
                  command=self.add_music_files).pack(fill=tk.X, pady=2)
        ttk.Button(music_button_frame, text="Remove", 
                  command=self.remove_music_file).pack(fill=tk.X, pady=2)
        ttk.Button(music_button_frame, text="Clear All", 
                  command=self.clear_music_files).pack(fill=tk.X, pady=2)
        
        # Trial progress
        ttk.Label(main_frame, text="Trial Progress:").grid(row=3, column=0, sticky=tk.W)
        self.progress_var = tk.StringVar(value="0/0")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(row=3, column=1, sticky=tk.W)
        
        # Current status display
        self.status_frame = ttk.LabelFrame(main_frame, text="Current Status", padding="20")
        self.status_frame.grid(row=4, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(self.status_frame, text="Ready to start", 
                                     font=("Arial", 20, "bold"))
        self.status_label.pack()
        
        self.phase_label = ttk.Label(self.status_frame, text="", 
                                    font=("Arial", 14))
        self.phase_label.pack()
        
        self.timer_label = ttk.Label(self.status_frame, text="", 
                                    font=("Arial", 12))
        self.timer_label.pack()
        
        # Emotion rating section
        self.rating_frame = ttk.LabelFrame(main_frame, text="Emotion Rating (1-9)", padding="15")
        self.rating_frame.grid(row=5, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))
        
        # Rating scale
        rating_scale_frame = ttk.Frame(self.rating_frame)
        rating_scale_frame.pack()
        
        ttk.Label(rating_scale_frame, text="Negative").pack(side=tk.LEFT)
        
        self.rating_var = tk.IntVar(value=5)
        self.rating_scale = tk.Scale(rating_scale_frame, from_=1, to=9, 
                                    orient=tk.HORIZONTAL, variable=self.rating_var,
                                    length=300)
        self.rating_scale.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(rating_scale_frame, text="Positive").pack(side=tk.LEFT)
        
        # Rating labels
        rating_label_frame = ttk.Frame(self.rating_frame)
        rating_label_frame.pack(pady=10)
        
        ttk.Label(rating_label_frame, text="1-3: Negative", 
                 foreground="red").pack(side=tk.LEFT, padx=20)
        ttk.Label(rating_label_frame, text="4-6: Neutral", 
                 foreground="gray").pack(side=tk.LEFT, padx=20)
        ttk.Label(rating_label_frame, text="7-9: Positive", 
                 foreground="green").pack(side=tk.LEFT, padx=20)
        
        self.submit_rating_button = ttk.Button(self.rating_frame, text="Submit Rating", 
                                              command=self.submit_rating, state=tk.DISABLED)
        self.submit_rating_button.pack(pady=10)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=20)
        
        self.start_button = ttk.Button(button_frame, text="Start Experiment", 
                                      command=self.start_experiment)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Experiment", 
                                     command=self.stop_experiment, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(button_frame, text="Save Data", 
                                     command=self.save_data, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_bar_var = tk.StringVar(value="Ready - Add music files to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_bar_var, relief=tk.SUNKEN)
        status_bar.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def add_music_files(self):
        files = filedialog.askopenfilenames(
            title="Select Music Files",
            filetypes=[("Audio files", "*.mp3 *.wav *.ogg *.m4a"), ("All files", "*.*")]
        )
        
        for file in files:
            if file not in self.music_files:
                self.music_files.append(file)
                self.music_listbox.insert(tk.END, file.split('/')[-1])
                
        self.progress_var.set(f"0/{len(self.music_files)}")
        
    def remove_music_file(self):
        selection = self.music_listbox.curselection()
        if selection:
            index = selection[0]
            self.music_files.pop(index)
            self.music_listbox.delete(index)
            self.progress_var.set(f"0/{len(self.music_files)}")
            
    def clear_music_files(self):
        self.music_files.clear()
        self.music_listbox.delete(0, tk.END)
        self.progress_var.set("0/0")
        
    def start_experiment(self):
        self.participant_id = self.participant_entry.get().strip()
        if not self.participant_id:
            messagebox.showerror("Error", "Please enter participant ID")
            return
            
        if not self.music_files:
            messagebox.showerror("Error", "Please add music files")
            return
            
        # Shuffle music files
        np.random.shuffle(self.music_files)
        
        self.is_recording = True
        self.current_trial = 0
        self.data_buffer = []
        
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        
        self.acquisition.start_acquisition()
        self.status_bar_var.set("Experiment running")
        
        self._run_next_trial()
        
    def stop_experiment(self):
        self.is_recording = False
        pygame.mixer.music.stop()
        self.acquisition.stop_acquisition()
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL)
        self.submit_rating_button.config(state=tk.DISABLED)
        
        self.status_bar_var.set("Experiment stopped")
        self.status_label.config(text="Experiment stopped")
        self.phase_label.config(text="")
        self.timer_label.config(text="")
        
    def _run_next_trial(self):
        if not self.is_recording or self.current_trial >= len(self.music_files):
            self.stop_experiment()
            return
            
        self.current_music_file = self.music_files[self.current_trial]
        self.trial_start_time = time.time()
        self.current_phase = "music"
        self.recording_data = []
        
        self.progress_var.set(f"{self.current_trial + 1}/{len(self.music_files)}")
        
        # Update display
        self.status_label.config(text=f"Playing: {self.current_music_file.split('/')[-1]}")
        self.phase_label.config(text="Listen to the music")
        self.submit_rating_button.config(state=tk.DISABLED)
        
        # Start music
        try:
            pygame.mixer.music.load(self.current_music_file)
            pygame.mixer.music.play()
        except pygame.error as e:
            messagebox.showerror("Error", f"Cannot play music file: {e}")
            self._skip_trial()
            return
            
        # Schedule recording start
        self.root.after(int(self.recording_start * 1000), self._start_recording)
        
        # Schedule music end
        self.root.after(int(self.music_duration * 1000), self._end_music)
        
        # Update timer
        self._update_timer()
        
    def _start_recording(self):
        if not self.is_recording:
            return
            
        self.current_phase = "recording"
        self.phase_label.config(text="Recording EEG - Continue listening")
        
        # Start recording thread
        def record_data():
            start_time = time.time()
            while (time.time() - start_time) < self.recording_duration and self.is_recording:
                data1, data2 = self.acquisition.read_data()
                if data1 is not None:
                    current_time = time.time()
                    recording_time = current_time - start_time
                    
                    sample = {
                        'timestamp': current_time,
                        'recording_time': recording_time,
                        'channels': data1[:6]
                    }
                    self.recording_data.append(sample)
                    
                time.sleep(0.001)
                
        record_thread = threading.Thread(target=record_data)
        record_thread.start()
        
    def _end_music(self):
        if not self.is_recording:
            return
            
        pygame.mixer.music.stop()
        self.current_phase = "rating"
        
        self.status_label.config(text="Rate your emotional response")
        self.phase_label.config(text="How did the music make you feel?")
        self.timer_label.config(text="")
        
        self.rating_var.set(5)  # Reset to neutral
        self.submit_rating_button.config(state=tk.NORMAL)
        
    def submit_rating(self):
        if not self.is_recording:
            return
            
        rating = self.rating_var.get()
        
        # Determine emotion category
        if rating <= 3:
            emotion_category = "negative"
        elif rating <= 6:
            emotion_category = "neutral"
        else:
            emotion_category = "positive"
            
        # Store trial data
        trial_data = {
            'trial_id': self.current_trial + 1,
            'music_file': self.current_music_file,
            'participant_id': self.participant_id,
            'rating': rating,
            'emotion_category': emotion_category,
            'trial_start_time': self.trial_start_time,
            'samples': self.recording_data
        }
        
        self.data_buffer.append(trial_data)
        
        self.submit_rating_button.config(state=tk.DISABLED)
        self.current_trial += 1
        
        # Start rest period
        self._start_rest()
        
    def _start_rest(self):
        if not self.is_recording:
            return
            
        self.current_phase = "rest"
        self.status_label.config(text="Rest Period")
        self.phase_label.config(text="Take a break - relax")
        
        # Rest timer countdown
        def rest_countdown(remaining):
            if not self.is_recording:
                return
                
            self.timer_label.config(text=f"Next trial in: {remaining}s")
            
            if remaining > 0:
                self.root.after(1000, lambda: rest_countdown(remaining - 1))
            else:
                self._run_next_trial()
                
        rest_countdown(self.rest_duration)
        
    def _skip_trial(self):
        self.current_trial += 1
        self._run_next_trial()
        
    def _update_timer(self):
        if not self.is_recording or self.current_phase not in ["music", "recording"]:
            return
            
        elapsed = time.time() - self.trial_start_time
        
        if self.current_phase == "music":
            if elapsed < self.recording_start:
                remaining = self.recording_start - elapsed
                self.timer_label.config(text=f"Recording starts in: {remaining:.1f}s")
            else:
                remaining = self.music_duration - elapsed
                self.timer_label.config(text=f"Music ends in: {remaining:.1f}s")
        else:  # recording
            remaining = self.music_duration - elapsed
            self.timer_label.config(text=f"Recording... {remaining:.1f}s")
            
        if elapsed < self.music_duration:
            self.root.after(100, self._update_timer)
            
    def save_data(self):
        if not self.data_buffer:
            messagebox.showwarning("Warning", "No data to save")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_filename = f"emotion_data_{self.participant_id}_{timestamp}.json"
        
        session_info = {
            'participant_id': self.participant_id,
            'timestamp': timestamp,
            'sampling_rate': 250,
            'electrode_positions': ['F7', 'FT7', 'T7', 'F8', 'FT8', 'T8'],
            'protocol': {
                'music_duration': self.music_duration,
                'recording_start': self.recording_start,
                'recording_duration': self.recording_duration,
                'rest_duration': self.rest_duration
            }
        }
        
        data_to_save = {
            'session_info': session_info,
            'trials': self.data_buffer
        }
        
        with open(json_filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)
            
        # Save as CSV
        csv_filename = f"emotion_data_{self.participant_id}_{timestamp}.csv"
        
        with open(csv_filename, 'w', newline='') as f:
            fieldnames = ['trial_id', 'music_file', 'rating', 'emotion_category', 
                         'timestamp', 'recording_time', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for trial in self.data_buffer:
                for sample in trial['samples']:
                    row = {
                        'trial_id': trial['trial_id'],
                        'music_file': trial['music_file'].split('/')[-1],
                        'rating': trial['rating'],
                        'emotion_category': trial['emotion_category'],
                        'timestamp': sample['timestamp'],
                        'recording_time': sample['recording_time']
                    }
                    for i, ch_val in enumerate(sample['channels']):
                        row[f'ch{i+1}'] = ch_val
                    writer.writerow(row)
                    
        self.status_bar_var.set(f"Data saved: {json_filename}, {csv_filename}")
        messagebox.showinfo("Success", f"Data saved successfully:\n{json_filename}\n{csv_filename}")
        
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = EmotionAcquisitionGUI()
    app.run()
        csv_filename = f"emotion_data_{self.participant_id}_{timestamp}.csv"
        
        with open(csv_filename, 'w', newline='') as f:
            fieldnames = ['trial_id', 'music_file', 'rating', 'emotion_category', 
                         'timestamp', 'recording_time', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for trial in self.data_buffer:
                for sample in trial['samples']:
                    row = {
                        'trial_id': trial['trial_id'],
                        'music_file': trial['music_file'].split('/')[-1],
                        'rating': trial['rating'],
                        'emotion_category': trial['emotion_category'],
                        'timestamp': sample['timestamp'],
                        'recording_time': sample['recording_time']
                    }
                    for i, ch_val in enumerate(sample['channels']):
                        row[f'ch{i+1}'] = ch_val
                    writer.writerow(row)
                    
        self.status_bar_var.set(f"Data saved: {json_filename}, {csv_filename}")
        messagebox.showinfo("Success", f"Data saved successfully:\n{json_filename}\n{csv_filename}")
        
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = EmotionAcquisitionGUI()
    app.run()
