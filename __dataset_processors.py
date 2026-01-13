from __session_processors import session
from dynamic_access import SessionsWrapper


class dataset:
    def __init__(self, datasetName, sessionIds, signals, data_directory):
        self.datasetName = datasetName
        self.sessionIds = sessionIds
        self.signals = signals
        self.data_directory = data_directory

        self.sessions = SessionsWrapper({
            sid: session(sid, signals) for sid in sessionIds
        })



    def load_sessions(self, demographics_fp, **kwargs):
        for session in self.sessions.values():
            print(session.session_id)
            session.load_session(demographics_fp = demographics_fp, data_directory=self.data_directory)

    def preprocess_sessions(self, *args, **kwargs):
        for session in self.sessions.values():
            session.preprocess_signals(*args, **kwargs)
    
    def preprocess_data(self, *args, **kwargs):
        pass

    def get_quality_masks(self, *args, **kwargs):
        pass

    def get_activity_masks(self, *args, **kwargs):
        pass    
    
    def get_cardiac_cycles(self, *args, **kwargs):
        pass

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

