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
    
    # routes to specialized methods based on windowing type
    # specialized methods are implemented in subclasses
    def extract_fiducials(self, preprocessed, window_start: int, window_end: int, windowing_type: str):
        if windowing_type == "cardiac_cycles":
            return self.extract_cardiac_cycle_fiducials(preprocessed, window_start, window_end)
        elif windowing_type == "fixed_windows":
            return self.extract_window_fiducials(preprocessed, window_start, window_end)
        else:
            raise NotImplementedError(f"Fiducial extraction for windowing type '{windowing_type}' is not implemented.")


    def check_quality(self, preprocessed, window_start: int, window_end: int, fiducials: dict, windowing_type: str):
        if windowing_type == "cardiac_cycles":
            return self.check_quality_cardiac_cycle(preprocessed, window_start, window_end, fiducials)
        elif windowing_type == "fixed_windows":
            return self.check_quality_fixed_window(preprocessed, window_start, window_end, fiducials)
        else:
            raise NotImplementedError(f"Quality check for windowing type '{windowing_type}' is not implemented.")

