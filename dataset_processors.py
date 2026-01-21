from session_processors import Session
from dynamic_access import SessionsWrapper
import pandas as pd
import os
import numpy as np

class Dataset:
    def __init__(self, datasetName, sessionIds, signals, data_directory):
        self.datasetName = datasetName
        self.sessionIds = sessionIds
        self.signals = signals
        self.data_directory = data_directory
        self.windowing_type = None  # 'cardiac_cycles' or 'fixed_windows'

        self.sessions = SessionsWrapper({
            sid: Session(sid, signals, data_directory) for sid in sessionIds
        })


    def load_sessions(self, demographics_fp, **kwargs):
        for session in self.sessions.values():
            print(session.session_id)
            session.load_session(demographics_fp = demographics_fp)

    def preprocess_sessions(self, *args, **kwargs):
        for session in self.sessions.values():
            session.preprocess_signals(*args, **kwargs)
    
    def preprocess_data(self, *args, **kwargs):
        pass

    def get_quality_masks(self, *args, **kwargs):
        pass

    def get_activity_masks(self, *args, **kwargs):
        pass    
    
    # Run one of the following windowing methods before processing signals
    # can be cardiac cycles or fixed windows

    def get_cardiac_cycles(self, *args, **kwargs):
        """
        Iterates through all sessions and triggers cardiac cycle segmentation.
        
        Accepts any arguments required by the underlying session.detect_cycles() method.

        Stored in session objects as session.windows
        
        Common Kwargs:
            reference_signal_name (str): The name of the signal to use (default: "ECG_downsample").
            method (str): The algorithm to use (default: "r_peak").
        """        
        self.windowing_type = 'cardiac_cycles'
        for session in self.sessions.values():
            try:
                session.detect_cycles(*args, **kwargs)
            except Exception as e:
                print(f"Session {session.session_id}: Skipping cycle detection. Reason: {e}")
    
    def get_fixed_windows(self, length=10, **kwargs):
        """
        used instead of get cardiac cycles for fixed windowing approaches such as deep learning models
        will store indices in session.windows
        
        :param self: Dataset object
        :param length: int, length of each window in seconds
        :param kwargs: Description
        """
        self.windowing_type = 'fixed_windows'
        for session in self.sessions.values():
            try:
                session.get_fixed_windows(length=length, **kwargs)
            except Exception as e:
                print(f"Session {session.session_id}: Skipping fixed windowing. Reason: {e}")


    def annotate_cardiac_cycles_with_activities(self, *args, **kwargs):
        pass

    def compute_features(self, *args, **kwargs):
        pass

    def save_features(self, *args, **kwargs):
        pass

    def save_quality_masks(self, *args, **kwargs):
        pass

    def save_dataset(self, *args, **kwargs):
        pass

    def load_dataset(self, *args, **kwargs):
        pass

    def process_signals(self, signal_list: list[str]):
        """
        Traverses sessions, ensures cycles exist, processes signals, 
        and updates the fiducials CSV.
        """
        if self.windowing_type is None:
            raise RuntimeError("Windowing method not set. Please run get_cardiac_cycles() or get_fixed_windows() first.")
    
        for session in self.sessions.values():
            if session.windows is None or len(session.windows) == 0:
                print(f"Skipping session {session.session_id}")
                continue
            
            # create a list to hold results for all cycles in this session
            session_results = []
            for i, (start, end) in enumerate(session.windows):
                session_results.append({
                    "window_id": i,
                    "start": start,
                    "end": end
                })

            for signal_name in signal_list:
                signal = session.signals[signal_name]
                if not signal:
                    print(f"Signal {signal_name} not found in session.")
                    continue
                
                preprocessed_signal = signal.preprocess()

                for i, (start, end) in enumerate(session.windows):
                    fiducials = signal.extract_fiducials(preprocessed_signal, start, end, self.windowing_type)
                    is_good = signal.check_quality(preprocessed_signal, start, end, fiducials, self.windowing_type)
                    
                    session_results[i].update(fiducials)
                    session_results[i][f"{signal_name}_quality"] = is_good

            # Save results back to session fiducials CSV   
            if session_results:
                new_df = pd.DataFrame(session_results)
                session.save_fiducials(new_df)
            
            session.unload_session()
    
        '''
        Get features -> list of features and list of associated signals
        Calculate basic pwa features +on level of dataset, gets fiducials if not gotten
        This will demonstrate accessing the wave at the point of fiducial timing
        Basic PAT and PTT across signals
        Output to features csv which contains session, participant, window_id, feature value
        '''

    def compute_features(self, feature_map: dict):
        """
        Orchestrates feature calculation.
        Features can be one of pwa, pat, ptt.
        pat must contain ecg as one of the signals.
        ptt must contain digital stethoscope as one of the signals.
        """
        all_session_features = []

        for session in self.sessions.values():
            segments_path = os.path.join(self.data_directory, session.session_id, "segments.csv")
            if not os.path.exists(segments_path):
                print(f"Fiducials missing for {session.session_id}. Please run process_signals() first.")
                continue
            fiducial_df = pd.read_csv(segments_path)
            session.load_session() 

            for i, row_data in fiducial_df.iterrows():
                row = {
                    "participant_id": session.participant_id,
                    "session_id": session.session_id,
                    "window_id": row_data['window_id']
                }

                for func, params in feature_map.items():
                    if func.lower() == "pwa":
                        pwa = self.compute_pwa_features(session, row_data, signal_names=params)
                        row.update(pwa)
                    elif func.lower() == "pat":
                        features = self.compute_PAT(session, row_data, signal_names=params)
                        row.update(features)
                    elif func.lower() == "ptt":
                        features = self.compute_PTT(session, row_data, signal_names=params)
                        row.update(features)
                    else:
                        raise NotImplementedError(f"Feature computation '{func}' is not implemented.")
                all_session_features.append(row)

            session.unload_session()
        
        feature_df = pd.DataFrame(all_session_features)
        save_loc = os.path.join(self.data_directory, "features.csv")
        feature_df.to_csv(save_loc, index=False)
    
    def compute_pwa_features(self, session, row_data, signal_names):
        """
        Calculates Pulse Wave Analysis features from fiducials already in segments.csv
        Can only be run on cardiac cycle windowed data.

        """
        if self.windowing_type != 'cardiac_cycles':
            raise RuntimeError("PWA features can only be computed on cardiac cycle windowed data.")
        
        results = {}
        for signal_name in signal_names:
            try:
                if row_data[f"{signal_name}_quality"] == False:
                    results.update({
                    f"{signal_name}_amp": np.nan,
                    f"{signal_name}_amp_norm": np.nan,
                    f"{signal_name}_rise_time": np.nan
                    })
                    continue

                signal = session.signals[signal_name]
                data = signal.data
                
                # peak and foot values
                peak_idx = int(row_data[f"{signal_name}_peak"])
                foot_idx = int(row_data[f"{signal_name}_foot"])
                peak_val = data[peak_idx]
                foot_val = data[foot_idx]

                raw_amp = peak_val - foot_val
                window_data = data[int(row_data['start']):int(row_data['end'])]
                norm_amp = raw_amp / np.mean(window_data)

                results.update({
                    f"{signal_name}_amp": raw_amp,
                    f"{signal_name}_amp_norm": norm_amp,
                    f"{signal_name}_rise_time": (peak_idx - foot_idx) / signal.sampling_rate
                })
            except Exception as e:
                print(f"Error computing PWA features for session {session.session_id}, window {row_data["window_id"]}: {e}")
        
        return results

    def compute_PAT(self, session, row_data, signal_names):
        """
        Calculates PAT by comparing timing between two signals.
        Can only be run on cardiac cycle windowed data.
        ECG must be one of the signals.
        """
        if self.windowing_type != 'cardiac_cycles':
            raise RuntimeError("PAT can only be computed on cardiac cycle windowed data.")
        
        if "ECG_downsample" not in signal_names:
            raise ValueError("PAT computation requires 'ECG_downsample' as one of the signals.")
        
        results = {}

        for signal_name in signal_names:
            if signal_name != "ECG_downsample":
                if row_data[f"{signal_name}_quality"] == False:
                    results.update({f"PAT_{signal_name}": np.nan})
                    continue

                ecg_peak = int(row_data["ECG_downsample_R_peak"])

                if "ppg" in signal_name.lower():
                    ppg_foot = int(row_data[f"{signal_name}_foot"])
                    pat_value = (ppg_foot - ecg_peak)
                    results.update({f"PAT_{signal_name}": pat_value})
                elif "earbud" in signal_name.lower():
                    earbud_a = int(row_data[f"{signal_name}_a"])
                    pat_value = (earbud_a - ecg_peak)
                    results.update({f"PAT_{signal_name}": pat_value})
                else:
                    print(f"Signal {signal_name} not recognized for PAT computation.")
        
        return results