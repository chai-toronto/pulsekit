from .base_signal import PhysiologicalSignal
from .signal_processing_utils import apply_butterworth_filter
from scipy.signal import find_peaks, hilbert, welch
import numpy as np
from scipy.stats import pearsonr

class Stethoscope(PhysiologicalSignal):
    def preprocess(self):
        processed = apply_butterworth_filter(self.data, 40, "high", fs=self.sampling_rate)
        return apply_butterworth_filter(processed, 240, "low", fs=self.sampling_rate)

    def extract_cardiac_cycle_fiducials(self, preprocessed, window_start: int, window_end: int):
        pcg_segment = preprocessed[window_start + 10 : window_start + 150]
        
        if len(pcg_segment) == 0:
             return {}

        pcg_height_req = np.mean(np.abs(pcg_segment)) + np.std(np.abs(pcg_segment))
        pcg_peaks, _ = find_peaks(pcg_segment, distance=30, height=pcg_height_req)
        
        if len(pcg_peaks) > 0:
            return {f"{self.name}_s1": window_start + 10 + pcg_peaks[0]}
        
        return {}

    def check_quality_cardiac_cycle(self, preprocessed, window_start: int, window_end: int, fiducials: dict):
        """
        Validates the cardiac cycle by ensuring exactly two distinct peaks (S1 and S2) 
        exist within the ECG R-R interval envelope.
        """
        segment = preprocessed[window_start:window_end]
        if len(segment) < 10: return False

        envelope = np.abs(hilbert(segment))
        
        # Find all peaks above noise floor, pick top two (80 ms distance)
        all_peaks, properties = find_peaks(envelope, distance=int(0.08 * self.sampling_rate), height=np.mean(envelope))        
        if len(all_peaks) < 2:
            return False
        heights = properties['peak_heights']
        sorted_indices = np.argsort(heights)[::-1]
        primary_h = heights[sorted_indices[0]]
        secondary_h = heights[sorted_indices[1]]
        
        # if more than two peaks, get the 3rd, if 3rd peak is almost as tall as S1, there's not truly two peaks
        if len(all_peaks) > 2:
            tertiary_h = heights[sorted_indices[2]]
            if tertiary_h > (secondary_h * 0.7):
                return False
        
        # Relative Saliency Check: S1 must still be at least 15% of the max peak to ensure its not noise
        if (secondary_h / primary_h) < 0.15:
            return False

        # s1 within 200ms of r-peak, s2 after 200
        top_two_peaks = sorted([all_peaks[sorted_indices[0]], all_peaks[sorted_indices[1]]])
        sound_split = int(0.2 * self.sampling_rate)
        if top_two_peaks[0] > sound_split or top_two_peaks[1] < int(0.2 * self.sampling_rate):
            return False

        return True
    
    def check_quality_fixed_window(self, preprocessed, window_start: int, window_end: int, fiducials: dict) -> float:
        '''
        Calculates a Signal Quality Index (SQI) for a fixed PCG window.
        Uses a fused-envelope approach to ensure S1 and S2 remain paired, 
        anchors strictly on the S1 onset, and correlates the envelopes to prevent phase mismatch.
        '''
        sig = preprocessed[window_start:window_end]
        fs = self.sampling_rate
        
        if len(sig) < fs or not np.isfinite(sig).all():
            return 0.0

        # Generate the Envelope
        env = np.abs(hilbert(sig))
        win_len = int(0.2 * fs)
        smoothed_env = np.convolve(env, np.ones(win_len)/win_len, mode='same')
        
    
        min_peak_dist = int(0.4 * fs)
        centers, _ = find_peaks(smoothed_env, distance=min_peak_dist, prominence=0.2 * np.mean(smoothed_env))
        
        s1_anchors = []
        for c in centers:
            search_start = max(0, c - int(0.25 * fs))
            # Widened search end to 0.20s to ensure the full hump is captured
            search_end = min(len(env), c + int(0.20 * fs)) 
            local_env = env[search_start:search_end]
            
            if len(local_env) == 0:
                continue
                
            # Find all distinct peaks in the hump without strict amplitude filtering yet
            local_peaks, _ = find_peaks(local_env, distance=int(0.08 * fs))
            
            # Keep peaks that are at least 25% of the loudest sound in this specific heartbeat
            valid_peaks = [p for p in local_peaks if local_env[p] > 0.25 * np.max(local_env)]
            
            if len(valid_peaks) > 0:
                s1_anchors.append(search_start + valid_peaks[0])
            else:
                s1_anchors.append(search_start + np.argmax(local_env))
                
        if len(s1_anchors) < 3:
            return 0.0
            
        # Beat Segmentation
        pre_samples = int(0.10 * fs)
        post_samples = int(0.50 * fs)

        beat_matrix = []
        for peak in s1_anchors:
            if peak - pre_samples < 0 or peak + post_samples >= len(env):
                continue
            
            # Correlating the Hilbert Envelope bypasses high-frequency phase shift errors
            beat_segment = env[peak - pre_samples : peak + post_samples]
            beat_matrix.append(beat_segment)
            
        beat_matrix = np.array(beat_matrix)
        
        if len(beat_matrix) < 3:
            return 0.0
            
        local_template = np.median(beat_matrix, axis=0)
        
        # Individual Beat Correlation
        correlations = []
        for beat in beat_matrix:
            r, _ = pearsonr(beat, local_template)
            
            if np.isnan(r): r = 0.0
            # Clip negative correlations to 0
            correlations.append(max(0.0, r))

        return float(np.mean(correlations))
    
    def extract_fixed_window_fiducials(self, preprocessed, window_start: int, window_end: int):
        """
        Extracts heart rate from stethoscope within the given window using both
        Power Spectral Density (PSD) and Inter-Beat Interval (IBI) peak detection.
        
        param self: stethoscope object
        param preprocessed: np.ndarray, preprocessed stethoscope signal data
        param window_start: int, index of window start
        param window_end: int, index of window end
        """
        from scipy.signal import welch, find_peaks, hilbert
        import numpy as np

        sig = preprocessed[window_start:window_end]
        fs = self.sampling_rate
        
        # Failsafe for empty windows
        if len(sig) == 0:
            return {f"{self.name}_HR": np.nan}
        
        # Generate the Envelope
        env = np.abs(hilbert(sig))
        win_len = int(0.2 * fs)
        smoothed_env = np.convolve(env, np.ones(win_len)/win_len, mode='same')
        
        # Locate the center of each fused heartbeat
        min_peak_dist = int(0.4 * fs)
        centers, _ = find_peaks(smoothed_env, distance=min_peak_dist, prominence=0.2 * np.mean(smoothed_env))
        
        s1_anchors = []
        for c in centers:
            search_start = max(0, c - int(0.25 * fs))
            search_end = min(len(env), c + int(0.20 * fs)) 
            local_env = env[search_start:search_end]
            
            if len(local_env) == 0:
                continue
                
            local_peaks, _ = find_peaks(local_env, distance=int(0.08 * fs))
            valid_peaks = [p for p in local_peaks if local_env[p] > 0.25 * np.max(local_env)]
            
            if len(valid_peaks) > 0:
                s1_anchors.append(search_start + valid_peaks[0])
            else:
                s1_anchors.append(search_start + np.argmax(local_env))
                
        # Calculate HR from IBI
        if len(s1_anchors) > 1:
            ibis_in_samples = np.diff(s1_anchors)
            median_ibi_samples = np.median(ibis_in_samples) 
            hr_bpm_ibi = float(60.0 * fs / median_ibi_samples)
        else:
            hr_bpm_ibi = np.nan

        return {
            f"{self.name}_HR": hr_bpm_ibi
        }