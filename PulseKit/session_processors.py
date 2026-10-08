from .signal_processors.base_signal import PhysiologicalSignal
from .dynamic_access import SignalsWrapper
import copy
import pandas as pd
import numpy as np
import os
from pathlib import Path


class Session:
    '''
    Represents a recording session for a participant.
    Loads demographics from a CSV file in the data directory.
    Manages physiological signals for the session.
    Only loads signal data into memory when needed.
    Loads physiological signals based on signal objects and those present in the session directory.

    :param self: Session object
    :param sessionId: str, unique identifier for the session
    :param signals: wrapped dictionary of PhysiologicalSignal objects to allow dot notation access
    :param data_directory: str, path to the main data directory
    :param windows: np.ndarray or None, start and end indices of windows in the session
    :param participant_id: str, identifier for the participant
    :param demographics: dict, participant demographics loaded from CSV
    :param height: float or None, participant height in cm
    :param weight: float or None, participant weight in kg
    :param age: int or None, participant age in years
    :param gender: str or None, participant gender, 1: male, 2: female
    :param length: int, length of the session in samples
    '''
    def __init__(self, sessionId, physiologicalSignals, data_directory):
        self.session_id = sessionId
        self.data_directory = os.path.abspath(data_directory)
        nested_records_directory = os.path.join(self.data_directory, "data")
        self.records_directory = (
            nested_records_directory
            if os.path.isdir(nested_records_directory)
            else self.data_directory
        )
        self.record_path = self._resolve_record_path(sessionId)
        # In nested WFDB layouts outputs live beside the record. For flat layouts,
        # keep outputs in the shared records directory and include the session ID.
        self.session_dir = os.path.dirname(self.record_path)
        self._flat_record_layout = self.session_dir == self.records_directory
        self.header = self._read_header()
        self.channel_indices = {
            name: index for index, name in enumerate(self.header.sig_name or [])
        }
        self._record = None
        self._channel_data = {}
        self.windows = None
        self.participant_id = self.get_session_id(sessionId)
        self.demographics = self.get_participant_demographics()
        
        self.height = self.demographics.get("Height")
        self.weight = self.demographics.get("Weight")
        self.age = self.demographics.get("Age")
        self.gender = self.demographics.get("Gender")


        # creating a session-specific copy of each signal object present in the session and passing in the directory
        signals_dict = {}
        
        for s in physiologicalSignals:
            if s.name in self.channel_indices:
                s_copy = copy.deepcopy(s)
                s_copy.session_dir = self.session_dir
                s_copy.sampling_rate = self.header.fs
                # Keep the existing signal.data interface: the first access loads
                # the WFDB record once at session scope and fans channels out.
                s_copy.load_data = self.load_session
                signals_dict[s.name] = s_copy
            else:
                print(f"Signal {s.name} is not present in WFDB record {sessionId}.")
        
        self.signals = SignalsWrapper(signals_dict)
        self.length = int(self.header.sig_len)

    def _resolve_record_path(self, session_id: str) -> str:
        """Find a WFDB record base path in nested or flat ``data`` layouts."""
        records_dir = Path(self.records_directory)
        candidates = [
            records_dir / session_id / session_id,
            records_dir / session_id,
        ]

        # RECORDS commonly uses paths like ``HC_10_1/HC_10_1``. Support a
        # record path supplied directly as the session ID as well.
        supplied = Path(session_id)
        if supplied.is_absolute():
            candidates.insert(0, supplied.with_suffix(""))
        else:
            candidates.insert(0, records_dir / supplied.with_suffix(""))

        for candidate in candidates:
            if candidate.with_suffix(".hea").is_file():
                return str(candidate)

        attempted = ", ".join(str(path.with_suffix(".hea")) for path in candidates)
        raise FileNotFoundError(
            f"WFDB header for session '{session_id}' was not found. Checked: {attempted}"
        )

    def _read_header(self):
        try:
            import wfdb
        except ImportError as exc:
            raise ImportError(
                "WFDB input requires the 'wfdb' package. Install it with 'pip install wfdb'."
            ) from exc
        return wfdb.rdheader(self.record_path)
        
    def get_session_id(self, dir_) -> str:
        '''
        Get participant ID from directory name.
        Handles formats like HC_33, HFC_33, and strips sub-session suffixes (e.g., HC_33_2 -> HC_33).
        '''
        # Check if there are 2 or more underscores (e.g., HC_33_2 or HFC_45_3)
        if dir_.count('_') >= 2:
            # Split exactly once from the right (the last underscore) and keep the front portion
            return dir_.rsplit('_', 1)[0]
        
        # If 1 or 0 underscores (e.g., "HC_33", "HFC_45", or "33"), return the whole string
        return dir_


    def get_participant_demographics(self) -> dict:
        '''
        Loads demographics from a CSV file located in the data directory.
        Expects columns: Participant, Height, Weight, Age
        Runs on initialization of Session object.
        
        :param self: Session object
        :return: dict with keys 'Height', 'Weight', 'Age', 'Gender'
        '''
        load_dir = os.path.join(self.data_directory, "demographics.csv")
        defaults = {
            "Height": None,
            "Weight": None,
            "Age": None,
            "Gender": None,
        }

        if not os.path.exists(load_dir):
            print(f"Warning: Demographics file not found at {load_dir}")
            return defaults

        try:
            df = pd.read_csv(load_dir)
            demo_tag = df['Participant'].astype(str).str.lower().str.replace("hfc", "").str.replace("hc", "").str.strip()
            dir_tag = self.participant_id.lower().replace("hfc_", "").replace("hc_", "")
            match_mask = demo_tag == dir_tag
            participant_row = df[match_mask]
            
            if not participant_row.empty:
                row = participant_row.iloc[0].to_dict()
                # Normalize Auricle's current column labels while retaining all
                # source fields in the demographics dictionary.
                row["Height"] = row.get("Height", row.get("Height (cm)"))
                row["Weight"] = row.get("Weight", row.get("Weight (kg)"))
                return row
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
        return self.length


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
        if self._record is None:
            import wfdb

            required_names = set(self.signals.keys())
            # These channels support existing BP quality checks and optional
            # activity labeling, even when not requested as processing signals.
            required_names.update(("Activity_Annotations", "CNAP_Annotations"))
            channels = [
                index for name, index in self.channel_indices.items()
                if name in required_names
            ]
            if not channels:
                raise ValueError(f"No requested signal channels found for {self.session_id}.")
            self._record = wfdb.rdrecord(
                self.record_path,
                physical=True,
                channels=channels,
            )
            self._channel_data = {
                name: np.asarray(self._record.p_signal[:, idx])
                for idx, name in enumerate(self._record.sig_name)
            }

        for signal in self.signals.values():
            if not signal.is_loaded:
                signal.data = self._channel_data[signal.name]

        cnap = self._channel_data.get("CNAP_Annotations")
        if cnap is not None:
            for signal in self.signals.values():
                if hasattr(signal, "_load_external_nibp_mask"):
                    signal.cnap_annotations = cnap
    
    def unload_session(self) -> None:
        '''
        Clears raw signal data from memory for all signals in this session.
        
        :param self: Session object
        '''
        for signal in self.signals.values():
            if hasattr(signal, 'unload_data'):
                signal.unload_data()
            if hasattr(signal, "cnap_annotations"):
                signal.cnap_annotations = None
        self._record = None
        self._channel_data = {}

    def get_channel(self, name: str):
        """Return a WFDB channel, loading the record on first access."""
        if name not in self.channel_indices:
            return None
        self.load_session()
        return self._channel_data.get(name)

    def get_segments_path(self, segments_filename: str = "segments.csv") -> str:
        """Return an output path that is unique in flat record layouts."""
        if self._flat_record_layout:
            stem, extension = os.path.splitext(segments_filename)
            return os.path.join(
                self.session_dir, f"{self.session_id}_{stem}{extension}"
            )
        return os.path.join(self.session_dir, segments_filename)

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
    
    def get_fixed_windows(self, length, overlap=0.0, **kwargs) -> None:
        '''
        Used instead of get cardiac cycles for fixed windowing approaches such as deep learning models.
        Stores indices in self.windows.
        
        :param length: int or float, length of each window in seconds.
        :param overlap: float, fraction of the window that overlaps with the previous one (0.0 to 0.99).
        :param kwargs: Additional arguments (not used).
        '''
        if not (0.0 <= overlap < 1.0):
            raise ValueError("Overlap must be between 0.0 (inclusive) and 1.0 (exclusive).")

        available_signals = list(self.signals.values())
        if not available_signals:
            raise ValueError("No signals available in this session to divide into windows.")

        first_signal = available_signals[0]
        
        sampling_rate = first_signal.sampling_rate
        window_size = int(length * sampling_rate)
        
        # Calculate how many samples to step forward for each new window
        step_size = int(window_size * (1.0 - overlap))
        
        # Fallback to prevent an infinite loop if window/overlap math results in a 0 step
        if step_size < 1:
            step_size = 1
        
        total_length = self.length
        
        windows = []
        # Advance by 'step_size' rather than 'window_size'
        for start in range(0, total_length - window_size + 1, step_size):
            end = start + window_size
            windows.append((start, end))
        
        self.windows = np.array(windows, dtype=np.int64)


    def save_fiducials(self, new_data_df: pd.DataFrame, segments_filename: str = "segments.csv") -> None:
        '''
        Saves or updates fiducials CSV file for this session.
        
        :param self: Session object
        :param new_data_df: new fiducials data to save
        :type new_data_df: pd.DataFrame
        '''
        file_path = self.get_segments_path(segments_filename)
        
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            new_cols = [c for c in new_data_df.columns if c != "window_id"]
            existing_df = existing_df.drop(columns=[c for c in new_cols if c in existing_df.columns], errors='ignore')
            final_df = pd.merge(existing_df, new_data_df, on="window_id", how="left")
        else:
            final_df = new_data_df
        
        final_df.to_csv(file_path, index=False)