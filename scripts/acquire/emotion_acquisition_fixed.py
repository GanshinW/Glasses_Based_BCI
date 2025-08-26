#!/usr/bin/env python3
"""
Emotion Acquisition (external audio): timer-only, record last 10s of 30s
- No internal music playback
- Clean CSV header, no '#' metadata lines
- Optional rating (can be skipped)
"""

import json
import csv
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import os
import numpy as np
from signal_acquisition import SignalAcquisition


class EmotionAcquisitionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion Recognition Data Acquisition (External Audio)")
        self.root.geometry("700x560")

        # dirs and session
        self.data_dir = os.path.join("data", "emotion")
        os.makedirs(self.data_dir, exist_ok=True)
        self.session_file = os.path.join(self.data_dir, "session_info.json")

        # acquisition
        self.acquisition = SignalAcquisition()
        self.is_recording = False
        self.participant_id = "P001"

        # timing config
        self.total_duration = 30.0        # seconds
        self.recording_start = 20.0       # seconds
        self.recording_duration = 10.0    # seconds

        # state
        self.trial_count = 0
        self.trial_start_time = 0.0
        self.current_phase = "idle"
        self.current_trial_data = []      # list of dict rows
        self.waiting_for_rating = False

        # rating (optional)
        self.rating_var = tk.IntVar(value=5)
        self.use_rating = True            # set False to disable rating UI

        self._setup_gui()
        self._load_session_info()

    def _setup_gui(self):
        main = ttk.Frame(self.root, padding="16")
        main.grid(row=0, column=0, sticky="nsew")

        # participant
        ttk.Label(main, text="Participant ID:").grid(row=0, column=0, sticky="w")
        self.participant_entry = ttk.Entry(main, width=18)
        self.participant_entry.insert(0, self.participant_id)
        self.participant_entry.grid(row=0, column=1, sticky="w", padx=(8, 0))

        # durations
        dfrm = ttk.LabelFrame(main, text="Timing (s)", padding="10")
        dfrm.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        ttk.Label(dfrm, text="Total").grid(row=0, column=0, padx=4)
        self.total_entry = ttk.Entry(dfrm, width=7)
        self.total_entry.insert(0, str(int(self.total_duration)))
        self.total_entry.grid(row=0, column=1)
        ttk.Label(dfrm, text="Record Start").grid(row=0, column=2, padx=10)
        self.start_entry = ttk.Entry(dfrm, width=7)
        self.start_entry.insert(0, str(int(self.recording_start)))
        self.start_entry.grid(row=0, column=3)
        ttk.Label(dfrm, text="Record Len").grid(row=0, column=4, padx=10)
        self.len_entry = ttk.Entry(dfrm, width=7)
        self.len_entry.insert(0, str(int(self.recording_duration)))
        self.len_entry.grid(row=0, column=5)

        # status
        sfrm = ttk.LabelFrame(main, text="Status", padding="12")
        sfrm.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 6))
        self.status_label = ttk.Label(sfrm, text="Ready. Start your external audio, then click 'Next Trial'.", font=("Arial", 12))
        self.status_label.pack()
        self.timer_label = ttk.Label(sfrm, text="", font=("Arial", 11))
        self.timer_label.pack(pady=(6, 0))

        # rating (optional)
        self.rating_frame = ttk.LabelFrame(main, text="Emotion Rating (1-9) [optional]", padding="10")
        self.rating_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 6))
        ttk.Label(self.rating_frame, text="Negative").pack(side=tk.LEFT)
        self.scale = tk.Scale(self.rating_frame, from_=1, to=9, orient=tk.HORIZONTAL, variable=self.rating_var, length=300)
        self.scale.pack(side=tk.LEFT, padx=10)
        ttk.Label(self.rating_frame, text="Positive").pack(side=tk.LEFT)
        if not self.use_rating:
            self.rating_frame.grid_remove()

        # buttons
        bfrm = ttk.Frame(main)
        bfrm.grid(row=4, column=0, columnspan=2, pady=(10, 0))
        self.next_btn = ttk.Button(bfrm, text="Next Trial", command=self.start_next_trial)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        self.submit_btn = ttk.Button(bfrm, text="Submit Rating / Save", command=self.submit_rating, state=tk.DISABLED)
        self.submit_btn.pack(side=tk.LEFT, padx=5)
        self.quit_btn = ttk.Button(bfrm, text="Quit", command=self.quit_application)
        self.quit_btn.pack(side=tk.LEFT, padx=5)

        # footer
        self.info_var = tk.StringVar(value="Ready")
        ttk.Label(main, textvariable=self.info_var).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        # layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

    def _load_session_info(self):
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'r') as f:
                    s = json.load(f)
                    self.trial_count = int(s.get('total_trials', 0))
            except Exception:
                pass

    def _update_session_info(self):
        # simple session summary
        session_info = {
            'task_type': 'emotion_recognition',
            'participant_id': self.participant_id,
            'last_updated': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'sampling_rate': 250,
            'electrode_positions': ['F7','FT7','T7','F8','FT8','T8'],
            'protocol': {
                'total_duration': self.total_duration,
                'recording_start': self.recording_start,
                'recording_duration': self.recording_duration
            },
            'total_trials': self.trial_count
        }
        with open(self.session_file, 'w') as f:
            json.dump(session_info, f, indent=2)

    def start_next_trial(self):
        # read UI params
        self.participant_id = self.participant_entry.get().strip() or "P001"
        try:
            self.total_duration = float(self.total_entry.get())
            self.recording_start = float(self.start_entry.get())
            self.recording_duration = float(self.len_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Timing values must be numbers")
            return

        if self.recording_start + self.recording_duration > self.total_duration:
            messagebox.showerror("Error", "recording_start + recording_duration must be <= total_duration")
            return

        # start DAQ if not started
        if not self.is_recording:
            self.acquisition.start_acquisition()
            self.is_recording = True

        # init trial
        self.trial_start_time = time.time()
        self.current_phase = "running"
        self.current_trial_data = []
        self.waiting_for_rating = False
        self.next_btn.config(state=tk.DISABLED)
        self.submit_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Trial running. Play external audio now.")
        self.info_var.set("Trial started")

        # schedule record start/end
        self.root.after(int(self.recording_start * 1000), self._start_recording)
        self.root.after(int(self.total_duration * 1000), self._end_trial)
        # update timer
        self._update_timer()

    def _start_recording(self):
        if self.current_phase != "running":
            return
        self.status_label.config(text="Recording EEG (last 10s window)")
        self._record_last_window()

    def _record_last_window(self):
        # record for recording_duration seconds in a background thread
        self.current_trial_data = []

        def worker():
            start_t = time.time()
            while time.time() - start_t < self.recording_duration and self.current_phase == "running":
                d1, d2 = self.acquisition.read_data()
                if d1 is not None:
                    now = time.time()
                    rec_t = now - start_t
                    # expect d1 has at least 6 channels
                    sample = {
                        'timestamp': now,
                        'recording_time': rec_t,
                        'ch1': d1[0], 'ch2': d1[1], 'ch3': d1[2],
                        'ch4': d1[3], 'ch5': d1[4], 'ch6': d1[5]
                    }
                    self.current_trial_data.append(sample)
                time.sleep(0.001)
        threading.Thread(target=worker, daemon=True).start()

    def _end_trial(self):
        if self.current_phase != "running":
            return
        self.current_phase = "rating" if self.use_rating else "done"
        self.status_label.config(text="Trial finished.")
        self.timer_label.config(text="")
        if self.use_rating:
            self.submit_btn.config(state=tk.NORMAL)
            self.waiting_for_rating = True
            self.info_var.set("Provide rating and click 'Submit Rating / Save'")
        else:
            # save immediately without rating
            self._save_current_trial(rating=None, category=None)
            self._reset_for_next()

    def submit_rating(self):
        if not self.waiting_for_rating:
            return
        rating = int(self.rating_var.get())
        # simple categorical mapping (optional)
        if rating <= 3:
            category = "negative"
        elif rating <= 6:
            category = "neutral"
        else:
            category = "positive"
        self._save_current_trial(rating=rating, category=category)
        self._reset_for_next()

    def _reset_for_next(self):
        self.trial_count += 1
        self._update_session_info()
        self.waiting_for_rating = False
        self.current_phase = "idle"
        self.next_btn.config(state=tk.NORMAL)
        self.submit_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Ready for next trial.")
        self.info_var.set("Saved. You can start the next trial.")

    def _update_timer(self):
        if self.current_phase != "running":
            return
        elapsed = time.time() - self.trial_start_time
        remain = max(0.0, self.total_duration - elapsed)
        self.timer_label.config(text=f"Elapsed: {elapsed:5.1f}s   Remaining: {remain:5.1f}s")
        if elapsed < self.total_duration:
            self.root.after(100, self._update_timer)

    def _save_current_trial(self, rating=None, category=None):
        # handle empty data
        if not self.current_trial_data:
            self.info_var.set("No data captured in last 10s window.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        # filename logic: keep clean; include rating/category if present
        base = f"emotion_trial_{self.participant_id}_{self.trial_count + 1}"
        if category is not None:
            base += f"_{category}"
        if rating is not None:
            base += f"_{int(rating)}"
        base += f"_{ts}.csv"
        csv_path = os.path.join(self.data_dir, base)

        # clean CSV: header only
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['timestamp', 'recording_time', 'ch1','ch2','ch3','ch4','ch5','ch6']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in self.current_trial_data:
                row = {
                    'timestamp': float(s.get('timestamp', 0.0)),
                    'recording_time': float(s.get('recording_time', 0.0)),
                    'ch1': float(s.get('ch1', 0.0)),
                    'ch2': float(s.get('ch2', 0.0)),
                    'ch3': float(s.get('ch3', 0.0)),
                    'ch4': float(s.get('ch4', 0.0)),
                    'ch5': float(s.get('ch5', 0.0)),
                    'ch6': float(s.get('ch6', 0.0)),
                }
                writer.writerow(row)

        self.info_var.set(f"Trial saved: {os.path.basename(csv_path)}")

    def quit_application(self):
        if self.is_recording:
            self.acquisition.stop_acquisition()
        self.root.quit()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = EmotionAcquisitionGUI()
    app.run()
