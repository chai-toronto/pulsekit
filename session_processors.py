from signal_processors.base_signal import PhysiologicalSignal
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
        self.length = self.get_session_length()
        

    def get_participant_demographics(self) -> dict:
        '''
        Loads demographics from a CSV file located in the data directory.
        Expects columns: Participant, Height, Weight, Age
        Runs on initialization of Session object.
        
        :param self: Session object
        :return: dict with keys 'Height', 'Weight', 'Age', 'Gender'
        '''
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
        
    def get_session_length(self) -> int:
        '''
        Determines the length of the session in samples.
        This length should be the same for every signal in the session.
        
        :param self: Session object
        :return: int, length of session in samples
        '''
        first_signal = next(iter(self.signals.values()))
        length = len(first_signal.data)
        first_signal.unload_data()
        return length


    def get_signal(self, name: str) -> PhysiologicalSignal:
        '''
        Retrieves a physiological signal object by name.
        
        :param self: Session object
        :param name: str, name of the signal to retrieve
        :return: PhysiologicalSignal or None
        '''
        return self.signals.get(name, None)
    
    def load_session(self) -> None:
        '''
        Loads raw signal data into memory for all signals in this session.
        
        :param self: Session object
        '''
        for signal in self.signals.values():
            signal.load_data()
    
    def unload_session(self) -> None:
        '''
        Clears raw signal data from memory for all signals in this session.
        
        :param self: Session object
        '''
        for signal in self.signals.values():
            if hasattr(signal, 'unload_data'):
                signal.unload_data()

    def detect_cycles(self, reference_signal_name: str) -> None:
        '''
        Detects cardiac cycles using the specified reference signal.
        Stores start and end indices in self.windows.
        
        :param self: Session object
        :param reference_signal_name: str, name of the reference signal
        :type reference_signal_name: str
        '''
        try:
            ref_signal = getattr(self.signals, reference_signal_name)
        except AttributeError:
            raise ValueError(f"Reference signal '{reference_signal_name}' not found in Session {self.session_id}")
        
        if not hasattr(ref_signal, 'get_cardiac_cycles'):
            raise NotImplementedError(f"The signal type {type(ref_signal).__name__} does not have a get_cardiac_cycles method yet.")

        # run that reference signal's peak detection
        peaks = ref_signal.get_cardiac_cycles()

        if len(peaks) < 2:
            print(f"Warning: Not enough peaks found in {self.session_id}")
            self.windows = np.empty((0, 2), dtype=np.int64)
            return
        
        self.windows = np.column_stack((peaks[:-1], peaks[1:]))
    
    def get_fixed_windows(self, length, **kwargs) -> None:
        '''
        Used instead of get cardiac cycles for fixed windowing approaches such as deep learning models
        Stores indices in self.windows
        
        :param self: Session object
        :param length: int, length of each window in seconds
        :param kwargs: Additional arguments (not used)
        '''
        signal = self.signals["ECG_downsample"]
        sampling_rate = signal.sampling_rate
        window_size = int(length * sampling_rate)
        
        total_length = signal.data.shape[0]
        
        windows = []
        for start in range(0, total_length - window_size + 1, window_size):
            end = start + window_size
            windows.append((start, end))
        
        self.windows = np.array(windows, dtype=np.int64)


    def save_fiducials(self, new_data_df: pd.DataFrame) -> None:
        '''
        Saves or updates fiducials CSV file for this session.
        
        :param self: Session object
        :param new_data_df: new fiducials data to save
        :type new_data_df: pd.DataFrame
        '''
        file_path = os.path.join(self.data_directory, self.session_id, "segments.csv")
        
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            new_cols = [c for c in new_data_df.columns if c != "window_id"]
            existing_df = existing_df.drop(columns=[c for c in new_cols if c in existing_df.columns], errors='ignore')
            final_df = pd.merge(existing_df, new_data_df, on="window_id", how="left")
        else:
            final_df = new_data_df
        
        final_df.to_csv(file_path, index=False)