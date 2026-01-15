# functions for extracting fiducial points from each signal type so that the specific functions can be easily found and updated
import numpy as np
from scipy.signal import find_peaks
from .signal_processing_utils import extract_fiducial_points

def extract_ppg(signal_data: np.ndarray, beat_start: int, beat_end: int, fs: int = 1000):
    """
    Extracts PPG fiducials using foot-to-foot segmentation.
    Returns None if fewer than 2 feet are found.
    """
    # Create a buffer window similar to legacy code (start to end + 500ms)
    search_window_end = min(len(signal_data), beat_end + 500)
    clip = signal_data[beat_start:search_window_end]
    
    # Flip and find first two peaks (which correspond to the signal feet)
    flipped_clip = -clip
    flipped_mean = np.mean(flipped_clip)
    feet, _ = find_peaks(flipped_clip, height=flipped_mean, distance=200)
    
    if len(feet) < 2:
        return None

    foot1 = feet[0]
    foot2 = feet[1]

    # Find the systolic peak (head) between the two feet
    # Note: relative to the start of the clip
    head_idx = np.argmax(signal_data[beat_start + foot1 : beat_start + foot2])
    
    # Segment foot-to-foot wave with 200ms buffer in front for derivative calc
    # These indices are relative to the full signal
    segment_start_idx = max(0, beat_start + foot1 - 200)
    segment_end_idx = beat_start + foot2
    segment = signal_data[segment_start_idx : segment_end_idx]

    # Double derivative to get acceleration wave
    ppg_derivative = np.diff(segment)
    ppg_second_deriv = np.diff(ppg_derivative)

    # Find 'a' peak on the second derivative
    a_peak_rel = np.argmax(ppg_second_deriv)
    
    # Extract standard fiducial points (a, b, c, d, e, f...)
    # We pass c_peak (which is 'a' here) as the anchor
    points = extract_fiducial_points(ppg_second_deriv, c_peak=a_peak_rel)
    
    # Package results with absolute indices
    results = {
        "peak": beat_start + foot1 + head_idx,
        "foot": beat_start + foot1,
        "foot_next": beat_start + foot2,
        # Points returned by extract_fiducial_points are relative to the segment start
        # We add segment_start_idx to map them back to global indices
        "a": points[0] + segment_start_idx,
        "b": points[1] + segment_start_idx,
        "c": points[2] + segment_start_idx,
        "d": points[3] + segment_start_idx,
        "e": points[4] + segment_start_idx,
        "f": points[5] + segment_start_idx,
        "z": points[6] + segment_start_idx,
        "q": points[7] + segment_start_idx,
        "g": points[8] + segment_start_idx
    }
    return results

def extract_earbud(signal_data: np.ndarray, beat_start: int, beat_end: int, fs: int = 1000):
    """
    Extracts Earbud fiducials using a theoretical arrival window.
    """
    # Theoretical arrival window parameters (100ms to 500ms after R-peak)
    theoretical_min = int(0.1 * 0.3 * fs) - 10 # 1/10 s/m * 0.3 m * 1000 samples/s (subtract 10 for buffer)
    theoretical_max = 500 
    
    # Ensure window doesn't go out of bounds
    win_start = beat_start + theoretical_min
    win_end = min(beat_start + theoretical_max, len(signal_data))
    
    arrival_window = signal_data[win_start : win_end]
    
    if len(arrival_window) == 0:
        return None

    # Find max peak in theoretical window
    a_peak_global = np.argmax(arrival_window) + win_start
    
    # Define segment for extraction (Start - 200ms to End + 200ms)
    segment_start_idx = max(0, beat_start - 200)
    segment_end_idx = min(len(signal_data), beat_end + 200)
    segment = signal_data[segment_start_idx : segment_end_idx]
    
    # Calculate anchor relative to the segment start
    anchor_rel = a_peak_global - segment_start_idx
    
    # Extract points
    points = extract_fiducial_points(segment, c_peak=anchor_rel)
    
    results = {
        "a": points[0] + segment_start_idx,
        "b": points[1] + segment_start_idx,
        "c": points[2] + segment_start_idx,
        "d": points[3] + segment_start_idx,
        "e": points[4] + segment_start_idx,
        "f": points[5] + segment_start_idx,
        "z": points[6] + segment_start_idx,
        "q": points[7] + segment_start_idx,
        "g": points[8] + segment_start_idx
    }
    return results

def extract_accelerometer(signal_data: np.ndarray, beat_start: int, beat_end: int, fs: int = 1000):
    # Placeholder for Accel specific extraction 
    return {"AO": "N/A"}

def extract_stethoscope(signal_data: np.ndarray, beat_start: int, beat_end: int, fs: int = 1000):
    pcg_segment = signal_data[beat_start + 10 : beat_start + 150]
    pcg_height_req = np.mean(np.abs(pcg_segment)) + np.std(np.abs(pcg_segment))
    pcg_peaks, _ = find_peaks(pcg_segment, distance=30, height=pcg_height_req)
    
    s1 = None
    if len(pcg_peaks) > 0:
        s1 = beat_start + 10 + pcg_peaks[0]
        
    return {"s1": s1}

def extract_respiration_belt(signal_data: np.ndarray, beat_start: int, beat_end: int, fs: int = 1000):
    # Placeholder for Respiration Belt specific extraction 
    return {"RI": "N/A"}