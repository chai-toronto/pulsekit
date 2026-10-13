from .signal_processing_utils import apply_butterworth_filter
import numpy as np
from scipy.stats import kurtosis, pearsonr
from typing import Dict, Any
from .base_signal import PhysiologicalSignal
from scipy.signal import find_peaks, welch

class ECG(PhysiologicalSignal):
    def preprocess(self):
        processed = apply_butterworth_filter(self.data, 0.5, "high", fs=self.sampling_rate)
        return apply_butterworth_filter(processed, 40, "low", fs=self.sampling_rate)

    def get_cardiac_cycles(self):
        ecg_signal = self.data
        detection_signal = apply_butterworth_filter(ecg_signal, 5, "high", fs=self.sampling_rate)
        detection_signal = apply_butterworth_filter(detection_signal, 15, "low", fs=self.sampling_rate)
        squared_signal = detection_signal ** 2
        height_req = np.mean(squared_signal)
        rough_peaks, _ = find_peaks(squared_signal, distance=400, height=height_req)
        true_peaks = []
        search_window = int(0.05 * self.sampling_rate) # +/- 50ms window 
        
        for p in rough_peaks:
            start = max(0, p - search_window)
            end = min(len(ecg_signal), p + search_window)
            search_slice = ecg_signal[start:end]
            local_max_idx = np.argmax(search_slice)
            true_peak_idx = start + local_max_idx
            true_peaks.append(true_peak_idx)
            
        return np.array(true_peaks)

    def extract_cardiac_cycle_fiducials(self, preprocessed, window_start: int, window_end: int):
        return {f"{self.name}_R_peak": window_start}

    def check_quality_cardiac_cycle(self, preprocessed, window_start: int, window_end: int, fiducials: dict):
        window = preprocessed[window_start:window_end]
        if len(window) < int(0.3 * self.sampling_rate) or len(window) > int(1.5 * self.sampling_rate):
            return False
        
        try:
            tests = [
                (np.std(window) > 0.001), 
                (np.max(window[:100]) + np.max(window[-100:])) / 2 > np.mean(window), 
                (kurtosis(window, fisher=False) > 3.5)
            ]
            return all(tests)
        except Exception:
            return False

    
    def check_quality_fixed_window(self, preprocessed, window_start: int, window_end: int, fiducials: dict) -> float:
        '''
        Calculates a Signal Quality Index (SQI) for a fixed multi-beat window
        using Beat-by-Beat Adaptive Template Matching. Returns a float between 0.0 and 1.0.
        '''
        sig = preprocessed[window_start:window_end]
        
        # Local Peak Detection 
        try:
            detection_signal = apply_butterworth_filter(sig, 5, "high", fs=self.sampling_rate)
            detection_signal = apply_butterworth_filter(detection_signal, 15, "low", fs=self.sampling_rate)
        except ValueError:
            # Failsafe if window is too short for filter order
            return 0.0
            
        squared_signal = detection_signal ** 2
        height_req = np.mean(squared_signal)
        min_peak_dist = int(0.4 * self.sampling_rate) 
        rough_peaks, _ = find_peaks(squared_signal, distance=min_peak_dist, height=height_req)
        
        true_peaks = []
        search_window = int(0.05 * self.sampling_rate) # +/- 50ms window 
        
        for p in rough_peaks:
            start = max(0, p - search_window)
            end = min(len(sig), p + search_window)
            search_slice = sig[start:end]
            if len(search_slice) > 0:
                local_max_idx = np.argmax(search_slice)
                true_peaks.append(start + local_max_idx)

        # Require at least 3 valid beats to form a reliable consensus template
        if len(true_peaks) < 3:
            return 0.0
            
        # Beat Segmentation
        pre_samples = int(0.25 * self.sampling_rate)
        post_samples = int(0.40 * self.sampling_rate)
        
        beat_matrix = []
        for peak in true_peaks:
            # Discard beats that fall off the edge of the fixed window
            if peak - pre_samples < 0 or peak + post_samples >= len(sig):
                continue
            beat_segment = sig[peak - pre_samples : peak + post_samples]
            beat_matrix.append(beat_segment)
            
        beat_matrix = np.array(beat_matrix)
        
        # Need at least 3 strictly contained beats for the median template
        if len(beat_matrix) < 3:
            return 0.0
            
        # Generate Median Template
        local_template = np.median(beat_matrix, axis=0)
        
        # Individual Beat Correlation
        correlations = []
        for beat in beat_matrix:
            r, _ = pearsonr(beat, local_template)
            
            # Handle perfectly flat signals resulting in NaN correlation
            if np.isnan(r):
                r = 0.0
            # Clip negative correlations to 0 for SQI purposes
            correlations.append(max(0.0, r))
            
        # Window Aggregation
        window_sqi = float(np.mean(correlations))
        
        return window_sqi
    
    def extract_fixed_window_fiducials(self, preprocessed, window_start: int, window_end: int):

        sig = preprocessed[window_start:window_end]
        
        # Failsafe for empty windows
        if len(sig) == 0:
            return {f"{self.name}_HR": np.nan}

        # getting HR through IBI by finding peaks
        try:
            detection_signal = apply_butterworth_filter(sig, 5, "high", fs=self.sampling_rate)
            detection_signal = apply_butterworth_filter(detection_signal, 15, "low", fs=self.sampling_rate)
        except ValueError:
            # Failsafe if window is too short for filter order
            return 0.0
            
        squared_signal = detection_signal ** 2
        height_req = np.mean(squared_signal)
        min_peak_dist = int(0.4 * self.sampling_rate) 
        rough_peaks, _ = find_peaks(squared_signal, distance=min_peak_dist, height=height_req)
        
        true_peaks = []
        search_window = int(0.05 * self.sampling_rate) # +/- 50ms window 
        
        for p in rough_peaks:
            start = max(0, p - search_window)
            end = min(len(sig), p + search_window)
            search_slice = sig[start:end]
            if len(search_slice) > 0:
                local_max_idx = np.argmax(search_slice)
                true_peaks.append(start + local_max_idx)
        
        # Calculate Inter-Beat Intervals (IBIs) in seconds
        ibi_intervals = np.diff(true_peaks) / self.sampling_rate

        if len(ibi_intervals) > 0:
            avg_ibi = np.median(ibi_intervals)
            hr_bpm_ibi = 60.0 / avg_ibi
        else:
            hr_bpm_ibi = np.nan
        
        return {
            f"{self.name}_HR": hr_bpm_ibi
            }