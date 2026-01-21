from session_processors import Session
from dynamic_access import SessionsWrapper
import pandas as pd


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
                #try:
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

                #except Exception as e:
                #    print(f"Error processing {signal_name}: {e}")

            # Save results back to session fiducials CSV   
            if session_results:
                new_df = pd.DataFrame(session_results)
                session.save_fiducials(new_df)
            
            session.unload_session()
