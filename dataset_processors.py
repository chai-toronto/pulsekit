from session_processors import Session
from dynamic_access import SessionsWrapper
import pandas as pd
import os
import numpy as np
from feature_processors import Feature, PAT, PWA

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