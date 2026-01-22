from signal_processors.base import PhysiologicalSignal
from dynamic_access import SignalsWrapper
import copy
import pandas as pd
import numpy as np
import os


class Session:
    def __init__(self, sessionId, physiologicalSignals, data_directory):
        self.session_id = sessionId
        self.data_directory = data_directory
        self.windows = None
        self.physiologicalSignals = physiologicalSignals
        self.participant_id = sessionId.split("_")[0]
        demos = self.get_participant_demographics()
        
        self.height = demos.get("Height")
        self.weight = demos.get("Weight")
        self.age = demos.get("Age")
        self.gender = demos.get("Gender")

        # creating a session-specific copy of each signal object and passing in the directory
        signals_dict = {
            s.name: copy.deepcopy(s) for s in physiologicalSignals
        }
        for signal in signals_dict.values():
            signal.session_dir = os.path.join(data_directory, sessionId)
        self.signals = SignalsWrapper(signals_dict)
        

    def get_participant_demographics(self):
        load_dir = os.path.join(self.data_directory, "demographics.csv")
        defaults = {"Height": None, "Weight": None, "Age": None, "Gender": None}

        if not os.path.exists(load_dir):
            print(f"Warning: Demographics file not found at {load_dir}")
            return defaults

        try:
            df = pd.read_csv(load_dir)
            match_mask = df['Participant'].astype(str).str.lower().str.replace("rp", "").str.strip() == self.participant_id
            participant_row = df[match_mask]
            
            if not participant_row.empty:
                return participant_row.squeeze().to_dict()
            else:
                print(f"Warning: Participant {self.participant_id} not found in demographics file.")
                return defaults

        except Exception as e:
            print(f"Error reading demographics: {e}")
            return defaults

    def get_signal(self, name):
        return self.signals.get(name, None)
    
    def load_signals(self):
        return NotImplementedError("This method should be implemented in subclasses.")
    
    def load_session(self, *args, **kwargs):
        for signal in self.signals.values():
            signal.load_data()
    
    def unload_session(self):
        """
        Critical for memory management. 
        Drops raw signal data but keeps calculated attributes (like cardiac_cycles).
        """
        for signal in self.signals.values():
            if hasattr(signal, 'clear_data'):
                signal.clear_data()

    def detect_cycles(self, reference_signal_name: str):
        """
        Uses a reference signal to calculate beat-to-beat intervals.
        Stores result in self.cardiac_cycles as an (N, 2) int64 numpy array.
        Column 0: Start Index
        Column 1: End Index
        """
        try:
            ref_signal = getattr(self.signals, reference_signal_name)
        except AttributeError:
            raise ValueError(f"Reference signal '{reference_signal_name}' not found in Session {self.session_id}")

        # run that reference signal's peak detection
        peaks = ref_signal.find_peaks()

        if len(peaks) < 2:
            print(f"Warning: Not enough peaks found in {self.session_id}")
            self.windows = np.empty((0, 2), dtype=np.int64)
            return
        
        self.windows = np.column_stack((peaks[:-1], peaks[1:]))
    
    def get_fixed_windows(self, length, **kwargs):
        """
        used instead of get cardiac cycles for fixed windowing approaches such as deep learning models
        will store indices in self.windows
        
        :param length: int, length of each window in seconds
        :param kwargs: Description
        """
        signal = self.signals["ECG_downsample"]
        sampling_rate = signal.sampling_rate
        window_size = int(length * sampling_rate)
        
        total_length = signal.data.shape[0]
        
        windows = []
        for start in range(0, total_length - window_size + 1, window_size):
            end = start + window_size
            windows.append((start, end))
        
        self.windows = np.array(windows, dtype=np.int64)


    def save_fiducials(self, new_data_df):
        """
        Merges new fiducial columns into the existing CSV for this session.
        Matches on 'window_id'.
        """
        file_path = os.path.join(self.data_directory, self.session_id, "segments.csv")
        
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            new_cols = [c for c in new_data_df.columns if c != "window_id"]
            existing_df = existing_df.drop(columns=[c for c in new_cols if c in existing_df.columns], errors='ignore')
            final_df = pd.merge(existing_df, new_data_df, on="window_id", how="left")
        else:
            final_df = new_data_df
        
        final_df.to_csv(file_path, index=False)