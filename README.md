# **PulseKit**

PulseKit is a data processing library for use alongside the CrampeD dataset (LINK).
It enables a data pipleline to be built, which the user can select which signals it wishes to include.
The dataset object, when initialized, will create Session objects for each data session, and within each of those a specific instance of each PhysiologicalSignal object which you have included in your pipeline (one of PPG, ECG, DigitalStethscope, etc.).
The session objects will store information about the user, such as height, weight, age, gender, and participant_id.
The PhysiologicalSignal subclasses (PhysiologicalSignal, ECG, PPG, Earbud, Stethoscope, Accelerometer, RespirationBelt, BP, Hemodynamics) contain specificc signal processing code for the preprocessing, fiducial point extraction (if applicable), and quality checks that are applied across all sessions for these signals.

## **Initialization**




## **Customizing the Pipeline**

PulseKit allows you to override default signal processing behavior by "injecting" your own Python functions. This is useful when you have a specific sensor or research goal that requires a unique quality assessment or a different points of interest.

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
test_dataset.set_custom_quality_check("PPG_Finger_downsample", my_custom_quality_check)

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
test_dataset.set_custom_fiducial_extractor("ECG_downsample", my_custom_extractor)

```

## **How it Works**

1. **Scope**: When you use `test_dataset.set_custom_...`, the function is automatically propagated to that specific signal across **every session** currently loaded in your dataset.
2. **No Inheritance Required**: You do not need to create new classes or subclasses. You simply pass a standard Python function.
3. **Order of Execution**: During `process_signals()`, the pipeline checks if a "Custom" function exists. If it finds one, it runs that; otherwise, it falls back to the default built-in logic for that signal type.
