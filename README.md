# **PulseKit**

PulseKit is a data processing library for use alongside the Auricle dataset.
It enables a data pipeline to be built, and the user can select which signals they wish to include.
The dataset object, when initialized, will create Session objects for each data session, and within each of those, a specific instance of each PhysiologicalSignal object, which you have included in your pipeline (one of PPG, ECG, Digital Stethscope, etc.).
The session objects will store information about the subject, such as height, weight, age, gender, and participant_id.
The PhysiologicalSignal subclasses (PhysiologicalSignal, ECG, PPG, Earbud, Stethoscope, Accelerometer, RespirationBelt, BP, Hemodynamics) contain specific signal processing code for the preprocessing, fiducial point extraction (if applicable), and quality checks that are applied across all sessions for these signals.

## **Initialization**

To begin, define the signals you want to process and point the Dataset manager to the Auricle dataset root (the folder containing `demographics.csv` and `data/`). PulseKit uses WFDB session-level lazy loading: it reads each record's header when creating sessions, then loads the requested channels together from that session's `.dat` record only when processing starts. Install the dependencies with `pip install -r requirements.txt`.

```Python
from PulseKit import *

# 1. Define the signals you want to include in your pipeline
# Every signal is initialized with its name and sampling rate (Hz)
signals=[
    ECG(name="ECG"),
    PPG(name="PPG_Finger"),
    PPG(name="PPG_Ear"),
    Stethoscope(name="Digital_Stethoscope"),
    Earbud(name="Earbud_Left"),
    Earbud(name="Earbud_Right"),
    Accelerometer(name="Accelerometer_X"),
    Accelerometer(name="Accelerometer_Y"),
    Accelerometer(name="Accelerometer_Z"),
    RespirationBelt(name="Respiration_Belt"),
    BP(name="Diastolic_BP"),
    BP(name="Systolic_BP"),
    BP(name="NIBP"),
    Hemodynamics(name="Cardiac_Output"),
    Hemodynamics(name="Stroke_Volume"),
    Hemodynamics(name="SVR"),
    Hemodynamics(name="Heart_Rate")
]

# 2. Create the Dataset Manager
# Omit sessionIds to include every available record listed in RECORDS.
# Provide sessionIds to restrict processing to a selected subset.
test_dataset = Dataset(
    datasetName="MyBPProject",
    signals=signals,
    data_directory="path/to/Auricle_data"
)

# Optional subset:
# test_dataset = Dataset(
#     datasetName="MyBPProject",
#     sessionIds=["HC_41_1", "HC_41_2"],
#     signals=signals,
#     data_directory="path/to/Auricle_data"
# )

# 3. Define the windows (Cardiac Cycles or Fixed Length)
# This must be run before processing signals
test_dataset.get_cardiac_cycles(reference_signal_name="ECG")

# 4. Extract Fiducials and Quality Flags for signals of interest
test_dataset.process_signals(signal_list=["ECG", "PPG_Finger", "Systolic_BP"])
```
## **Machine Learning Example**

Once process_signals() is finished, PulseKit saves a segments.csv file in your data directory under each session. This file is the bridge between physiological signal processing and your ML model. We next run compute features with our features and targets, which will automatically filter cardiac cycles/windows by quality, so you can plug the values into your model.

```Python
dataset.compute_features(
    features=[
        PWA(dataset=dataset, signals=["PPG_Finger", "PPG_Ear"]),
        PAT(dataset=dataset, signals=["PPG_Finger", "Earbud_Left", "ECG"])
    ],
    targets=["Systolic_BP", "Diastolic_BP"],
    output_name="test_features"
)
```

An entire example of this pipeline can be seen in ptt_modelling.ipynb.

## **Customizing the Pipeline**

PulseKit allows you to override default signal processing behaviour by "injecting" your own Python functions. This is useful when you have a specific sensor or research goal that requires a unique quality assessment or different points of interest.

### **1. Custom Quality Check Template**

Use this when you want to change how a signal is marked as "Good" or "Bad" for a specific window.

```python
def my_custom_quality_check(signal_obj, preprocessed, start, end, fiducials, windowing_type):
    """
    Template for a custom Quality Check.
    
    Args:
        signal_obj: The actual Signal object (e.g., PPG, ECG).
        preprocessed (np.ndarray): The filtered signal data for the whole session.
        start (int): Start index of the current window.
        end (int): End index of the current window.
        fiducials (dict): Points already extracted (e.g., {'PPG_peak': 1050}).
        windowing_type (str): Either 'cardiac_cycles' or 'fixed_windows'.
        
    Returns:
        bool: True if the data is high quality, False if it should be rejected.
    """
    # Example: Reject if the signal is too flat (Peak-to-Peak is too low)
    segment = preprocessed[start:end]
    if (segment.max() - segment.min()) < 0.01:
        return False
        
    return True

# Apply to your dataset
test_dataset.set_custom_quality_check("PPG_Finger", my_custom_quality_check)

```

### **2. Custom Fiducial Extractor Template**

Use this when you want to use a different algorithm to find points of interest (like the dicrotic notch or R-peaks).

```python
def my_custom_extractor(signal_obj, preprocessed, start, end, windowing_type):
    """
    Template for a custom Fiducial Point Extractor.
    
    Args:
        signal_obj: The actual Signal object.
        preprocessed (np.ndarray): The filtered signal data for the whole session.
        start (int): Start index of the current window.
        end (int): End index of the current window.
        windowing_type (str): Either 'cardiac_cycles' or 'fixed_windows'.
        
    Returns:
        dict: A dictionary of points. Keys should follow the naming 
              convention "{signal_name}_{point_name}".
    """
    # Example: Simple max-point detection
    segment = preprocessed[start:end]
    local_max = segment.argmax()
    
    return {
        f"{signal_obj.name}_custom_peak": start + local_max
    }

# Apply to your dataset
test_dataset.set_custom_fiducial_extractor("ECG", my_custom_extractor)

```

## **How it Works**

1. **Scope**: When you use `test_dataset.set_custom_...`, the function is automatically propagated to that specific signal across **every session** currently loaded in your dataset.
2. **No Inheritance Required**: You do not need to create new classes or subclasses. You simply pass a standard Python function.
3. **Order of Execution**: During `process_signals()`, the pipeline checks if a "Custom" function exists. If it finds one, it runs that; otherwise, it falls back to the default built-in logic for that signal type.
