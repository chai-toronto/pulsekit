import numpy as np

class physiological_signal:
    """Base class for physiological signals."""

    def __init__(self, name, sampling_rate):
        self.name = name
        self.sampling_rate = sampling_rate
        self.peaks = []
        self.data = None

    def get_data(self):
        return self.data

    def get_sampling_rate(self):
        return self.sampling_rate
    
    def preprocess(self):
        """Placeholder method for preprocessing the signal."""
        pass

    def get_peaks(self):
        return self.peaks
    
    def set_peaks(self, peaks):
        self.peaks = peaks
    
    def get_activity_mask(self):
        """Placeholder method for getting activity mask."""
        pass

    def set_activity_mask(self, mask):
        """Placeholder method for setting activity mask."""
        pass

    def get_quality_mask(self):
        """Placeholder method for getting quality mask."""
        pass

    def load_data(self, session_id, signal_name, root_data_directory):
        load_dir = f"{root_data_directory}/{session_id}/"
        self.data = np.fromfile(f"{load_dir}{signal_name}", dtype = float)

