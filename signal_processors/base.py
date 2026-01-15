import numpy as np
import os
from typing import Dict, Any, Optional

class PhysiologicalSignal:
    """Base class for physiological signals."""

    def __init__(self, name, sampling_rate, session_dir=None):
        self.name = name
        self.sampling_rate = sampling_rate
        self.peaks = []
        self._data = None
        self.preprocessed_data = None
        self.session_dir = session_dir
        self.is_loaded = False
    
    @property
    def data(self):
        """
        The magic property. Accessing .data triggers this function.
        It ensures data is loaded before returning it.
        """
        if not self.is_loaded:
            self.load_data()    
        return self._data

    @data.setter
    def data(self, value):
        """Allows you to manually set .data = some_array if needed"""
        self._data = value
        self.is_loaded = (value is not None)

    def load_data(self):
        """
        Default loader for binary files. 
        Subclasses (like Audio) can override this if they use .wav or .csv
        """
        load_dir = os.path.join(self.session_dir, self.name)
        if os.path.exists(load_dir):
            self._data = np.fromfile(load_dir, dtype=float)
            self.is_loaded = True
        else:
            print(f"Warning: File not found at {load_dir}")

    def clear_data(self):
        """Frees memory by dropping the raw signal array."""
        self.data = None
        self.is_loaded = False

    def preprocess(self):
        """
        Base preprocess does nothing. 
        Subclasses should call super().preprocess() if needed.
        """
        pass

    def get_data(self):
        if not self.is_loaded:
            self.load_data()
        return self._data

    def get_sampling_rate(self):
        return self.sampling_rate

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
    
    def extract_fiducials(self, start_index, end_index):
        """
        Placeholder. Subclasses must implement this.
        Returns a dictionary: {'fiducial_name': index, ...}
        """
        return {}
    
    def check_quality(self, start_index, end_index, fiducials):
        """
        Returns True/False. 
        Subclasses implement specific logic (e.g. valid amplitude, SNR).
        """
        return True



