from .session_processors import Session
from .dynamic_access import SessionsWrapper
from .feature_processors import Feature
import pandas as pd
import os
import numpy as np


class Dataset:
    '''
    Dataset/pipeline manager for sessions and their respective physiological signals.
    Handles session loading, signal processing, feature computation, and saving results.

    :param self: Dataset object
    :param datasetName: str, name of the dataset
    :param sessionIds: list of str, unique identifiers for each session in the dataset
    :param signals: list of PhysiologicalSignal objects to be used in each session
    :param data_directory: str, path to the main data directory containing session data
    :param windowing_type: str or None, type of windowing used ('cardiac_cycles' or 'fixed_windows')
    :param sessions: wrapped dictionary of Session objects to allow dot notation access
    '''
    def __init__(self, datasetName, sessionIds, signals, data_directory):
        self.datasetName = datasetName
        self.sessionIds = sessionIds
        self.signals = signals
        self.data_directory = data_directory
        self.windowing_type = None  # 'cardiac_cycles' or 'fixed_windows'

        self.sessions = SessionsWrapper({
            sid: Session(sid, signals, data_directory) for sid in sessionIds
        })  
    
    # Run one of the following windowing methods before processing signals (cardiac cycles or fixed windows)

    def get_cardiac_cycles(self, *args, **kwargs) -> None:
        '''
        Detects cardiac cycles for all sessions using the specified reference signal.
        Stores start and end indices in each session's windows attribute.
        
        :param self: Dataset object
        :param args: Arguments to be passed to session objects
        :param kwargs: Keyword arguments to be passed to session objects
        '''    
        self.windowing_type = 'cardiac_cycles'
        for session in self.sessions.values():
            try:
                session.detect_cycles(*args, **kwargs)
            except Exception as e:
                print(f"Session {session.session_id}: Skipping cycle detection. Reason: {e}")
    
    def get_fixed_windows(self, length=10, **kwargs) -> None:
        """
        Used instead of get cardiac cycles for fixed windowing approaches such as deep learning models
        Will store indices in session.windows
        
        :param self: Dataset object
        :param length: int, length of each window in seconds
        :param kwargs: Additional keyword arguments to be passed to session objects
        """
        self.windowing_type = 'fixed_windows'
        for session in self.sessions.values():
            try:
                session.get_fixed_windows(length=length, **kwargs)
            except Exception as e:
                print(f"Session {session.session_id}: Skipping fixed windowing. Reason: {e}")

    def process_signals(self, signal_list: list[str]) -> None:
        '''
        Orchestrates signal preprocessing, fiducial extraction, and quality assessment.
        Stores results back to each session's fiducials CSV in the data directory.
        
        :param self: Dataset object
        :param signal_list: List of signal names to process
        :type signal_list: list[str]
        '''
        if self.windowing_type is None:
            raise RuntimeError("Windowing method not set. Please run get_cardiac_cycles() or get_fixed_windows() first.")
    
        for session in self.sessions.values():
            if session.windows is None or len(session.windows) == 0:
                print(f"Skipping session {session.session_id}")
                continue
            
            print(f"Processing session {session.session_id}...")

            # create a list to hold results for all cycles in this session
            session_results = []
            for i, (start, end) in enumerate(session.windows):
                session_results.append({
                    "window_id": i,
                    "start": start,
                    "end": end
                })

            for signal_name in signal_list:
                if signal_name not in session.signals:
                    print(f"Signal {signal_name} not found in session {session.session_id}.")
                    continue
                
                signal = session.signals[signal_name]
                
                preprocessed_signal = signal.preprocess()

                for i, (start, end) in enumerate(session.windows):
                    fiducials = signal.extract_fiducials(preprocessed_signal, start, end, self.windowing_type)
                    is_good = signal.check_quality(preprocessed_signal, start, end, fiducials, self.windowing_type)
                    
                    if fiducials is not None:
                        session_results[i].update(fiducials)
                    session_results[i][f"{signal_name}_quality"] = is_good

            # Save results back to session fiducials CSV   
            if session_results:
                new_df = pd.DataFrame(session_results)
                session.save_fiducials(new_df)
            
            session.unload_session()

    def compute_features(self, features: list[Feature], targets: list[str] = None, output_name: str = "features", activity_labels: bool = False) -> None:
        '''
        Computes features for all sessions and optionally filters based on target signal quality.
        
        :param self: Description
        :param features: List of Feature objects to compute.
        :type features: list[Feature]
        :param targets: List of signal names (strings) acting as ground truth/targets. 
                        If the quality of any target in this list is bad for a specific window,
                        all features calculated for that window will be set to NaN.
        :type targets: list[str]
        :param output_name: Name of the output CSV file to save features to (default: "features.csv")
        :type output_name: str
        :param activity_labels: If True, loads 'activity_mask' and labels each beat with the majority activity.
        :type activity_labels: bool
        '''
        all_session_features = []
        targets = targets or []

        for session in self.sessions.values():
            segments_path = os.path.join(self.data_directory, session.session_id, "segments.csv")
            
            if not os.path.exists(segments_path):
                print(f"Missing file at {segments_path}, skipping session {session.session_id}")
                continue
                
            fiducial_df = pd.read_csv(segments_path)
            
            # Warn if target quality columns are missing
            for t in targets:
                if f"{t}_quality" not in fiducial_df.columns:
                    print(f"WARNING: Target quality '{t}_quality' not found in {session.session_id}.")
            
            mask = None
            if activity_labels:
                mask_path = os.path.join(self.data_directory, session.session_id, "activity_mask")
                if os.path.exists(mask_path):
                    mask = np.fromfile(mask_path, dtype=np.uint8)

            session.load_session() 

            for i, row_data in fiducial_df.iterrows():
                row = {
                    "participant_id": session.participant_id,
                    "session_id": session.session_id,
                    "window_id": row_data['window_id']
                }

                for feature in features:
                    try:
                        feature_results = feature.compute(session, row_data)
                        row.update(feature_results)
                    except Exception as e:
                        print(f"Error computing {feature.NAME}: {e}")

                if targets:
                    for t in targets:
                        quality_col = f"{t}_quality"
                        
                        # Check if quality is good (default to False/Bad if column missing)
                        is_good = row_data.get(quality_col, False)

                        # Find all columns in segments.csv that belong to this target
                        target_cols = [col for col in row_data.index if col.startswith(t) and col != quality_col]
                        
                        for name_feat in target_cols:
                            if is_good:
                                row[name_feat] = row_data[name_feat]
                            else:
                                row[name_feat] = np.nan

                if activity_labels:
                    labels = {
                        0: "rest",
                        1: "breath_hold",
                        2: "valsalva",
                        3: "exercise",
                        4: "cold_pressor"
                    }
                    if mask is not None:
                        start_idx = int(row_data.get('start', 0)) 
                        end_idx = int(row_data.get('end', len(mask))) 
                        
                        # Boundary safety checks
                        start_idx = max(0, start_idx)
                        end_idx = min(len(mask), end_idx)
                        
                        if start_idx < end_idx:
                            beat_mask = mask[start_idx:end_idx]
                            majority_code = np.bincount(beat_mask).argmax()
                            row['activity'] = labels.get(majority_code, "unknown")
                        else:
                            row['activity'] = "invalid_interval"
                    else:
                        row['activity'] = np.NaN

                all_session_features.append(row)

            session.unload_session()
        
        if not all_session_features:
            print("WARNING: no features computed. Check your input data/paths.")
            return

        feature_df = pd.DataFrame(all_session_features)
        
        save_loc = os.path.join(self.data_directory, f"{output_name}.csv")
        feature_df.to_csv(save_loc, index=False)
        print(f"Saved to: {save_loc}")

    def set_custom_quality_check(self, signal_name: str, func: callable):
        """
        Overrides the quality check for a specific signal across all sessions.
        The function must accept: (signal_obj, preprocessed, start, end, fiducials, windowing_type)
        Note: This function should return a boolean indicating quality. It also may or may not rely on fiducials.

        :param self: Dataset object
        :param signal_name: str, name of the signal to override
        :param func: callable, custom quality check function
        """
        count = 0
        for session in self.sessions.values():
            if signal_name in session.signals:
                session.signals[signal_name]._custom_quality_check = func
                count += 1
        print(f"Applied custom quality check to {signal_name} across {count} sessions.")

    def set_custom_fiducial_extractor(self, signal_name: str, func: callable):
        """
        Overrides the fiducial extractor for a specific signal across all sessions.
        The function must accept: (signal_obj, preprocessed, start, end, windowing_type)

        :param self: Dataset object
        :param signal_name: str, name of the signal to override
        :param func: callable, custom fiducial extraction function
        """
        count = 0
        for session in self.sessions.values():
            if signal_name in session.signals:
                session.signals[signal_name]._custom_fiducial_extractor = func
                count += 1
        print(f"Applied custom extractor to {signal_name} across {count} sessions.")


    # TODO: Review with ken, original methods which have not been implemented
    
    def annotate_cardiac_cycles_with_activities(self, *args, **kwargs):
        pass

    def save_features(self, *args, **kwargs):
        pass

    def save_quality_masks(self, *args, **kwargs):
        pass

    def save_dataset(self, *args, **kwargs):
        pass

    def load_dataset(self, *args, **kwargs):
        pass

    def load_sessions(self):
        for session in self.sessions.values():
            print(session.session_id)
            session.load_session()

    def preprocess_sessions(self, *args, **kwargs):
        for session in self.sessions.values():
            session.preprocess_signals(*args, **kwargs)
    
    def preprocess_data(self, *args, **kwargs):
        pass

    def get_quality_masks(self, *args, **kwargs):
        pass

    def get_activity_masks(self, *args, **kwargs):
        pass  