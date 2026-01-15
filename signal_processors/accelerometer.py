from .base import PhysiologicalSignal
from .signal_processing_utils import apply_butterworth_filter
import numpy as np

class Accelerometer(PhysiologicalSignal):
    def preprocess(self):
        super().preprocess()
        return apply_butterworth_filter(self.data, 0.5, "high", fs=self.sampling_rate)

    def extract_fiducials(self, preprocessed, beat_start: int, beat_end: int):
        return {f"{self.name}_AO": "N/A"}

    def check_quality(self, preprocessed, beat_start: int, beat_end: int, fiducials: dict):
        window = preprocessed[beat_start:beat_end]
        return (np.std(window) > 0.01)