from .base_signal import PhysiologicalSignal
from .signal_processing_utils import apply_butterworth_filter, extract_fiducial_points
from scipy.signal import find_peaks, welch, hilbert
import numpy as np
from scipy.stats import pearsonr

class Earbud(PhysiologicalSignal):
    def preprocess(self):
        processed = apply_butterworth_filter(self.data, 0.5, "high", fs=self.sampling_rate)
        return apply_butterworth_filter(processed, 15, "low", fs=self.sampling_rate)

    def extract_cardiac_cycle_fiducials(self, preprocessed, window_start: int, window_end: int):

        theoretical_min = int(0.1 * 0.3 * self.sampling_rate) - 10 
        theoretical_max = 300 
        
        win_start = window_start + theoretical_min
        win_end = min(window_start + theoretical_max, len(preprocessed))
        arrival_window = preprocessed[win_start : win_end]
        
        if len(arrival_window) == 0:
            return {}

        a_peak_global = np.argmax(arrival_window) + win_start
        
        segment_start_idx = max(0, window_start - 200)
        segment_end_idx = min(len(preprocessed), window_end + 200)
        segment = preprocessed[segment_start_idx : segment_end_idx]
        
        anchor_rel = a_peak_global - segment_start_idx
        points = extract_fiducial_points(segment, c_peak=anchor_rel)
        
        return {
            f"{self.name}_a": points[0] + segment_start_idx,
            f"{self.name}_b": points[1] + segment_start_idx,
            f"{self.name}_c": points[2] + segment_start_idx,
            f"{self.name}_d": points[3] + segment_start_idx,
            f"{self.name}_e": points[4] + segment_start_idx,
            f"{self.name}_f": points[5] + segment_start_idx,
            f"{self.name}_z": points[6] + segment_start_idx,
            f"{self.name}_q": points[7] + segment_start_idx,
            f"{self.name}_g": points[8] + segment_start_idx
        }

    def check_quality_cardiac_cycle(self, preprocessed, window_start: int, window_end: int, fiducials: dict):
        if not fiducials or f'{self.name}_a' not in fiducials:
            return False

        # Define QC window: Peak - 300 to Peak + 700
        peak_loc = fiducials[f'{self.name}_a']
        qc_start = max(0, peak_loc - 300)
        qc_end = min(len(preprocessed), peak_loc + 700)
        qc_window = preprocessed[qc_start:qc_end]
        
        if len(qc_window) == 0:
            return False

        # Peak relative to window start (is 300 unless we clipped at start)
        relative_peak_loc = 300 if peak_loc >= 300 else peak_loc

        try:
            qc_points = extract_fiducial_points(qc_window, c_peak=relative_peak_loc)
            point_amps = [qc_window[x] if x < len(qc_window) else 0 for x in qc_points]
            
            tests = [
                np.argmax(point_amps) in [0, 2],
                np.argmin(point_amps) == 1,
                (np.max(qc_window[:qc_points[2]]) - np.min(qc_window[:qc_points[2]]) * .5) > 
                (np.max(qc_window[qc_points[2]:]) - np.min(qc_window[qc_points[2]:])),
                qc_points[2] - qc_points[1] < 150,
            ]
            return all(tests)
        except Exception:
            return False

    
    def check_quality_fixed_window(self, preprocessed, window_start: int, window_end: int, fiducials: dict) -> float:
        '''
        Calculates a Signal Quality Index (SQI) for a fixed multi-beat window
        using Beat-by-Beat Adaptive Template Matching. Returns a float between 0.0 and 1.0.
        Anchors directly on the Earbud 'a peak' (local maxima).
        '''
        sig = preprocessed[window_start:window_end]
        fs = self.sampling_rate
        
        # Failsafe for extremely short windows
        if len(sig) < fs:
            return 0.0
        
        env = np.abs(hilbert(sig))
        win_len = int(fs * 1.5)
        local_mean = np.convolve(env, np.ones(win_len)/win_len, mode='same')
        
        dynamic_threshold = local_mean * 1.5 + 1e-4
        min_peak_dist = int(0.4 * fs) # Assumes max HR ~150 BPM
        
        # Detect energy bursts (complex centers)
        complex_centers, _ = find_peaks(env, height=dynamic_threshold, distance=min_peak_dist)
        
        a_peaks = []
        search_radius = int(0.15 * fs) # Search +/- 150ms around the energy burst
        
        for center in complex_centers:
            start = max(0, center - search_radius)
            end = min(len(sig), center + search_radius)
            local_region = sig[start:end]
            
            if len(local_region) > 0:
                a_peaks.append(start + np.argmax(local_region))
        
        # Require at least 3 valid beats to form a reliable consensus template
        if len(a_peaks) < 3:
            return 0.0
            
        pre_samples = int(0.30 * fs)
        post_samples = int(0.70 * fs)
        
        beat_matrix = []
        for peak in a_peaks:
            # Discard beats that fall off the edge of the fixed window
            if peak - pre_samples < 0 or peak + post_samples >= len(sig):
                continue
            beat_segment = sig[peak - pre_samples : peak + post_samples]
            beat_matrix.append(beat_segment)
            
        beat_matrix = np.array(beat_matrix)
        
        if len(beat_matrix) < 3:
            return 0.0
            
        # Generate Median Template
        local_template = np.median(beat_matrix, axis=0)
        
        # Individual Beat Correlation
        correlations = []
        for beat in beat_matrix:
            r, _ = pearsonr(beat, local_template)
            
            if np.isnan(r):
                r = 0.0
            # Clip negative correlations to 0
            correlations.append(max(0.0, r))
            
        # Window Aggregation
        window_sqi = float(np.mean(correlations))
        
        return window_sqi
    
    def extract_fixed_window_fiducials(self, preprocessed, window_start: int, window_end: int):
        """
        Extracts heart rate from earbud within the given window using both
        Power Spectral Density (PSD) and Inter-Beat Interval (IBI) peak detection.
        """
        from scipy.signal import welch, find_peaks, hilbert
        import numpy as np

        sig = preprocessed[window_start:window_end]
        fs = self.sampling_rate
        
        # Failsafe for empty windows
        if len(sig) == 0:
            return {f"{self.name}_HR": np.nan}
            
        env = np.abs(hilbert(sig))
        
        # Create a 1.5 second moving average to establish the local baseline energy
        win_len = int(fs * 1.5)
        local_mean = np.convolve(env, np.ones(win_len)/win_len, mode='same')
        
        dynamic_threshold = local_mean * 1.5 + 1e-4
        min_peak_dist = int(0.4 * fs) # Assumes max HR ~150 BPM
        
        complex_centers, _ = find_peaks(env, height=dynamic_threshold, distance=min_peak_dist)
        
        true_peaks = []
        search_radius = int(0.15 * fs)
        
        for center in complex_centers:
            start = max(0, center - search_radius)
            end = min(len(sig), center + search_radius)
            local_region = sig[start:end]
            
            if len(local_region) == 0:
                continue
                
            local_peaks, _ = find_peaks(local_region, distance=int(0.05 * fs), prominence=0.2 * np.mean(np.abs(local_region)))
            
            if len(local_peaks) > 0:
                best_peak_idx = local_peaks[np.argmax(local_region[local_peaks])]
                true_peaks.append(start + best_peak_idx)
            else:
                true_peaks.append(start + np.argmax(local_region))
            
        # Calculate HR from IBI
        if len(true_peaks) > 1:
            ibis_in_samples = np.diff(true_peaks)
            median_ibi_samples = np.median(ibis_in_samples)
            hr_bpm_ibi = float(60.0 * fs / median_ibi_samples)
        else:
            hr_bpm_ibi = np.nan

        return {
            f"{self.name}_HR": hr_bpm_ibi
        }
    