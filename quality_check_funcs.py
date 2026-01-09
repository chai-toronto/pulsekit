# Quality checks for each signal type so that the specific functions can be easily found and updated
import numpy as np
from pulsekit.signal_processing_utils import apply_butterworth_filter, extract_fiducial_points
from scipy.stats import kurtosis

def check_earbud(earbud_window, **kwargs):
    a_peak = kwargs["peak"]
    fiducial_points = extract_fiducial_points(earbud_window, c_peak=a_peak)
    fiducial_point_amplitudes = [earbud_window[x] if x < len(earbud_window) else 0 for x in fiducial_points]
    try:
        tests = [
            np.argmax(fiducial_point_amplitudes) in [0,2],
            np.argmin(fiducial_point_amplitudes) == 1,
            np.max(earbud_window[:fiducial_points[2]])-np.min(earbud_window[:fiducial_points[2]])*.5 > np.max(earbud_window[fiducial_points[2]:])-np.min(earbud_window[fiducial_points[2]:]),
            fiducial_points[2]-fiducial_points[1] < 150,
        ]
    except Exception as e:
        tests = [False, False, False, False]

    return all(tests)

def check_ppg(window, fs=1000, **kwargs):
    window = np.diff(window, prepend=window[0])
    window = np.diff(window, prepend=window[0])
    window = apply_butterworth_filter(window, 25, "low", fs=fs)

    a_peak = np.argmax(window[:int(window.shape[0]/2)])
    fiducial_points = extract_fiducial_points(window, c_peak=a_peak)
    fiducial_point_amplitudes = [window[x] if x < len(window) else 0 for x in fiducial_points]
    try:
        tests = [
            np.argmax(fiducial_point_amplitudes) in [0,2],
            np.argmin(fiducial_point_amplitudes) == 1,
            np.max(window[:fiducial_points[2]])-np.min(window[:fiducial_points[2]])*.5 > np.max(window[fiducial_points[2]:])-np.min(window[fiducial_points[2]:]),
            fiducial_points[2]-fiducial_points[1] < 150,
        ]
    except Exception as e:
        tests = [False, False, False, False]

    return all(tests)

def check_ecg(window, **kwargs):
    tests = [
        (np.std(window) > 0.001), # sensor zero
        (np.max(window[:100])+np.max(window[-100:]))/2 > np.mean(window),
        (kurtosis(window, fisher=False) > 3.5)
    ]
    return all(tests)

def check_accelerometer_X(window, **kwargs):
    tests = [
        (np.std(window) > 0.01), # sensor zero
    ]
    return all(tests)

def check_accelerometer_Y(window, **kwargs):
    tests = [
        (np.std(window) > 0.01), # sensor zero
    ]
    return all(tests)

def check_accelerometer_Z(window, **kwargs):
    tests = [
        (np.std(window) > 0.01), # sensor zero
    ]
    return all(tests)

def check_stethoscope(window, **kwargs):
    tests = [
        (np.std(window) > 0.001), # sensor zero
    ]
    return all(tests)

def check_respiration_belt(window, **kwargs):
    tests = [
        (np.std(window) > 0.01), # sensor zero
    ]
    return all(tests)

#bp
#uses mask, applies to beats
# option to return the mask too