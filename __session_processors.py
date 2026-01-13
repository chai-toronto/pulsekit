from __signal_processors import physiological_signal
from dynamic_access import SignalsWrapper
import copy


class session:
    def __init__(self, sessionId, physiologicalSignals):
        self.session_id = sessionId
        self.age = None
        self.height = None
        self.weight = None
        self.gender = None
        self.physiologicalSignals = physiologicalSignals
        self.signals = SignalsWrapper({physiologicalSignal.name: copy.deepcopy(physiologicalSignal) for physiologicalSignal in physiologicalSignals})

    def get_participant_demographics(self, demographics_fp):
        """placeholder method for getting participant demographics."""
        return {"height": 170, "age": 30, "weight": 70, "gender": "unknown"}

    def get_signal(self, name):
        return self.signals.get(name, None)
    
    def load_signals(self):
        return NotImplementedError("This method should be implemented in subclasses.")
    
    def load_session(self, *args, **kwargs):
        demographics = self.get_participant_demographics(demographics_fp=kwargs.get('demographics_fp', None))
        self.height = demographics.get('height', None)
        self.age = demographics.get('age', None)
        self.weight = demographics.get('weight', None)
        self.gender = demographics.get('gender', None)

        for signal in self.signals.values():
            signal.load_data(self.session_id, signal.name, kwargs.get('data_directory', None))