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

    def compute_features(self, features: list[Feature]) -> None:
        '''
        Computes features for all sessions based on previously extracted fiducials.
        Stores results back to a features CSV in the data directory.
        Necessary signals and windowing type are validated in Feature subclasses.
        
        :param self: Dataset object
        :param features: List of Feature objects to compute
        :type features: list[Feature]
        '''
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

                for feature in features:
                    try:
                        feature_results = feature.compute(session, row_data)
                        row.update(feature_results)
                    except Exception as e:
                        print(f"Error computing {feature.NAME} for session {session.session_id}, window {row_data['window_id']}: {e}")
                all_session_features.append(row)

            session.unload_session()
        
        feature_df = pd.DataFrame(all_session_features)
        save_loc = os.path.join(self.data_directory, "features.csv")
        feature_df.to_csv(save_loc, index=False)

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