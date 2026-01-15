import numpy as np
from .signal_processing_utils import ecg_peaks
#from acq_utils import save_channel_data
from typing import Tuple, List, Literal, Dict, Optional
import os
import pandas as pd
from pulsekit import preprocessing_funcs
from pulsekit import quality_check_funcs
from pulsekit import fiducial_extraction_funcs

class PulseKit():
    '''
        data_loc: directory location of the acq data files
        fs: frequency to downsample to (default 1000 Hz)
    '''
    def __init__(self, data_loc: str, fs: int = 1000):
        self.fs = fs
        self.data_loc = data_loc
        if not os.path.isdir(data_loc):
            raise ValueError("Data directory does not exist")
        else:
            self.sessions = os.listdir(data_loc) # list of all session locations in the dataset
        

    def segment(self, method: str = "ECG_R_peak"):
        for session_tag in self.sessions:
            # load in path for the session
            session_path = os.path.join(self.data_loc, session_tag)
            if not os.path.isdir(session_path):
                continue
            
            # based on method that is given, get that data and run the method (for now only ECG R-peak)
            if method.lower() == "ecg_r_peak":
                ecg_data = None
                for f in os.listdir(session_path):
                    if "ecg_downsample" in f.lower():
                        path = os.path.join(session_path, f)            
                        ecg_data = np.fromfile(path, dtype=np.float64)
                if ecg_data is not None:
                    peaks = ecg_peaks(ecg_data)
                else:
                    raise ValueError("No ECG data found for session {session_tag}")
            else:
                raise ValueError(
                    "no matches found for peak finding method given"
                )
            
            # saves beats in file after they have been extracted
            peaks = np.asarray(peaks)
            df_beats = pd.DataFrame({
                "cycle_id": np.arange(len(peaks) - 1),
                "beat_start": peaks[:-1],
                "beat_end": peaks[1:]
            })
            df_beats.to_csv(os.path.join(session_path, "segmented_beats.csv"), index=False)


    # signals list can contain any of the following: ["PPG_Finger", "PPG_Ear", "PPG_Wrist", "Earbud_Left", "Earbud_Right", "ECG", "Digital_Stethoscope", "AccX", "AccY", "AccZ"]
    # segmentation must be run before
    # will extract fiducial points for each beat in each session for the given signals, and append quality check results for that signal
    def process_signal(self, signals: List[str]):
        # check the signal provided against possible signals
        possible_signals = ["PPG_Finger", "PPG_Ear", "PPG_Wrist", "Earbud_Left", "Earbud_Right", "ECG", "Digital_Stethoscope", "AccX", "AccY", "AccZ"]
        for signal in signals:
            if signal not in possible_signals:
                raise ValueError(f"Signal {signal} not recognized. Possible signals are: {possible_signals}")
        
        # traverse each session and run quality checks on each beat for the given signals
        for session_tag in self.sessions:
            # load in path for the session
            session_path = os.path.join(self.data_loc, session_tag)
            if not os.path.isdir(session_path):
                continue

            # load the data for that session
            data = {}
            for f in os.listdir(session_path):
                path = os.path.join(session_path, f)
                name = f.replace("_downsample", "")
                data[name] = np.fromfile(path, dtype=np.float64)
            
            # load in segmented beats
            beat_path = os.path.join(session_path, "segmented_beats.csv")
            if not os.path.isfile(beat_path):
                print(f"No segmented beats found for session {session_tag}, skipping quality check.")
                continue

            df_beats = pd.read_csv(beat_path)
            beats = df_beats.to_numpy()

            for signal in signals:
                signal_results = []
                raw_signal = data[signal]
                preprocessed_signal = self.preprocess_signal(signal, raw_signal)

                # Iterate through beats
                for beat_row in beats:
                    cycle_id = int(beat_row[0])
                    start = int(beat_row[1])
                    end = int(beat_row[2])

                    # Extract Fiducials
                    fiducials = self.run_extract_fiducial_points(signal, preprocessed_signal, start, end)

                    # Initialize result row
                    row_data = {
                        "cycle_id": cycle_id,
                        "beat_start": start,
                        "beat_end": end,
                        "quality_check_passed": False # Default
                    }

                    if fiducials:
                        row_data |= fiducials
                        
                        # first we slice the preprocessed signal to the relevant window for quality checking
                        
                        # checking earbud in a window around 'a' peak
                        if "earbud" in signal.lower():
                            qc_start = fiducials['a'] - 300
                            qc_end = fiducials['a'] + 700
                            qc_kw = {"peak": 300} # Peak is relative to window start
                        elif "ppg" in signal.lower():
                            qc_start = fiducials['foot']
                            qc_end = fiducials['foot_next']
                            qc_kw = {}
                        else:
                            # Fallback for others (ECG, etc)
                            qc_start = start
                            qc_end = end
                            qc_kw = {}

                        # Bound checks
                        qc_start = max(0, qc_start)
                        qc_end = min(len(preprocessed_signal), qc_end)
                        qc_window = preprocessed_signal[qc_start:qc_end]
                        
                        # run quality check function
                        qc_result = self.run_quality_check(signal, qc_window, **qc_kw)
                        row_data["quality_check_passed"] = qc_result
                    else:
                        # Fiducial extraction failed (e.g. both feet not found in PPG)
                        row_data["quality_check_passed"] = False
                    
                    signal_results.append(row_data)

                if signal_results:
                        df = pd.DataFrame(signal_results)
                        filename = f"{signal}_processed.csv"
                        df.to_csv(os.path.join(session_path, filename), index=False)
                

    def preprocess_signal(self, signal: str, data: np.ndarray) -> np.ndarray:
        signal_map = {
            "ppg_finger": "preprocess_ppg",
            "ppg_ear": "preprocess_ppg",
            "ppg_wrist": "preprocess_ppg",
            "earbud_left": "preprocess_earbud",
            "earbud_right": "preprocess_earbud",
            "ecg": "preprocess_ecg",
            "digital_stethoscope": "preprocess_stethoscope",
            "accx": "preprocess_accelerometer",
            "accy": "preprocess_accelerometer",
            "accz": "preprocess_accelerometer"
        }
        signal_key = signal.lower()
        func_name = signal_map.get(signal_key)

        if func_name and hasattr(preprocessing_funcs, func_name):
            func = getattr(preprocessing_funcs, func_name)
            return func(data, fs=self.fs)
        else:
            print(f"No preprocessing function mapped for signal {signal}, returning original data.")
            return data
        
    def run_quality_check(self, signal: str, window: np.ndarray, **kwargs) -> bool:
        signal_map = {
            "ppg_finger": "check_ppg",
            "ppg_ear": "check_ppg",
            "ppg_wrist": "check_ppg",
            "earbud_left": "check_earbud",
            "earbud_right": "check_earbud",
            "ecg": "check_ecg",
            "digital_stethoscope": "check_stethoscope",
            "accx": "check_accelerometer_X",
            "accy": "check_accelerometer_Y",
            "accz": "check_accelerometer_Z"
        }
        signal_key = signal.lower()
        func_name = signal_map.get(signal_key)

        if func_name and hasattr(quality_check_funcs, func_name):
            func = getattr(quality_check_funcs, func_name)
            return func(window, fs=self.fs, **kwargs)
        else:
            print(f"No quality check function mapped for signal {signal}, returning False.")
            return False

    def run_extract_fiducial_points(self, signal: str, data: np.ndarray, start: int, end: int) -> dict:
            signal_map = {
                "ppg_finger": "extract_ppg",
                "ppg_ear": "extract_ppg",
                "ppg_wrist": "extract_ppg",
                "earbud_left": "extract_earbud",
                "earbud_right": "extract_earbud",
                "ecg": "extract_ecg",
                "digital_stethoscope": "extract_stethoscope",
                "accx": "extract_accelerometer",
                "accy": "extract_accelerometer",
                "accz": "extract_accelerometer"
            }

            func_name = signal_map.get(signal.lower())
            if func_name and hasattr(fiducial_extraction_funcs, func_name):
                func = getattr(fiducial_extraction_funcs, func_name)
                return func(data, start, end, fs=self.fs)
            else:
                print(f"No extraction function mapped for signal {signal}.")
                return None
    
    # extract sbp and dbp and filter invalid readings 
    def process_bp(self):
        return None

    def extract_features(self, features: List[str]):
        # check if quality check has been run, if so, only extract features from beat ids which passed
        # if no quality check has been run, extract features from all beats


        return None
    
    # ability to append activity data to existing fiducial files


