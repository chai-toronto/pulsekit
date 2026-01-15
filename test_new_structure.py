import sys
import os
import pandas as pd
import numpy as np
from signal_processors import ECG, PPG, Stethoscope, Earbud
from __dataset_processors import Dataset              
from signal_processing_utils import ecg_peaks  

print("--- 1. Initializing Dataset ---")

# Create the Manager
test_dataset = Dataset(
    datasetName="TestDataset",
    sessionIds=["1", "3"],
    signals=[
        ECG(name="ECG_downsample", sampling_rate=1000),
        PPG(name="PPG_Finger_downsample", sampling_rate=1000),
        PPG(name="PPG_Wrist_downsample", sampling_rate=1000),
        Stethoscope(name="Digital_Stethoscope_downsample", sampling_rate=1000),
        Earbud(name="Earbud_Left_downsample", sampling_rate=1000)
    ],
    data_directory="test_data"
)

# Load the raw data into memory
test_dataset.load_sessions(demographics_fp="test_data/demographics.csv")

# Verify loading using Dot Notation
print(f"Session 1 Loaded. ECG Shape: {test_dataset.sessions._1.signals.ECG_downsample.data.shape}")
print(f"Session 3 Loaded. PPG Finger Shape: {test_dataset.sessions._3.signals.PPG_Finger_downsample.data.shape}")

'''
# ==============================================================================
# 2. SEGMENTATION (Replaces: pipe.segment())
# ==============================================================================

print("\n--- 2. Segmenting Beats (ECG R-Peaks) ---")

# We iterate through sessions and manually trigger the segmentation logic
# (In a finished app, you would move this logic inside test_dataset.get_cardiac_cycles())

for session in test_dataset.sessions.values():
    # A. Access the ECG Object
    ecg_obj = session.signals.ECG_downsample
    
    # B. Preprocess (Filter 0.5-40Hz)
    ecg_obj.preprocess()
    
    # C. Find Peaks (Reusing your existing util)
    # Note: We use the raw data from the object
    r_peaks = ecg_peaks(ecg_obj.data)
    ecg_obj.set_peaks(r_peaks) # Store them back in the object if needed
    
    # D. Create the Beat DataFrame (Start -> End)
    # Convert list to array for slicing
    r_peaks = np.array(r_peaks)
    
    df_beats = pd.DataFrame({
        "cycle_id": np.arange(len(r_peaks) - 1),
        "beat_start": r_peaks[:-1],
        "beat_end": r_peaks[1:]
    })
    
    # E. Save to disk (To match old functionality)
    save_path = os.path.join(test_dataset.data_directory, session.session_id, "segmented_beats.csv")
    df_beats.to_csv(save_path, index=False)
    
    # Store df in session object for immediate use in next step
    session.beat_table = df_beats
    print(f"Saved {len(df_beats)} beats for Session {session.session_id}")

# Verification
rp1_beats = pd.read_csv("test_data/1/segmented_beats.csv")
print(rp1_beats.head())


# ==============================================================================
# 3. PROCESS PPG FINGER (Replaces: pipe.process_signal(["PPG_Finger"]))
# ==============================================================================
print("\n--- 3. Processing PPG Finger ---")

# Define which signals to process
target_signals = ["PPG_Finger_downsample"] 

for session in test_dataset.sessions.values():
    # Load the beats we just created
    beats = session.beat_table.to_numpy()
    
    for sig_name in target_signals:
        # A. Dynamic Access (using your wrapper logic)
        # Note: getattr handles the fuzzy matching (PPG_Finger matches PPG_Finger_downsample)
        signal_obj = getattr(session.signals, sig_name)
        
        # B. Preprocess
        signal_obj.preprocess()
        
        # C. Beat-by-Beat Analysis
        results = []
        for beat in beats:
            cycle_id, start, end = int(beat[0]), int(beat[1]), int(beat[2])
            
            # This calls the method we wrote in the previous turn!
            beat_res = signal_obj.analyze_beat(start, end)
            
            # Add metadata
            beat_res['cycle_id'] = cycle_id
            beat_res['beat_start'] = start
            beat_res['beat_end'] = end
            results.append(beat_res)
            
        # D. Save Results
        df_res = pd.DataFrame(results)
        save_path = os.path.join(test_dataset.data_directory, session.session_id, "PPG_Finger_processed.csv")
        df_res.to_csv(save_path, index=False)

# Verification
rp1_finger = pd.read_csv("test_data/1/PPG_Finger_processed.csv")
print(rp1_finger.head())


# ==============================================================================
# 4. PROCESS OTHERS (Replaces: pipe.process_signal(["Wrist", "Steth", "Earbud"]))
# ==============================================================================
print("\n--- 4. Processing Wrist, Stethoscope, Earbud ---")

target_signals = ["PPG_Wrist_downsample", "Digital_Stethoscope_downsample", "Earbud_Left_downsample"]

for session in test_dataset.sessions.values():
    beats = session.beat_table.to_numpy()
    
    for sig_name in target_signals:
        # A. Access Object
        # Note: If your wrapper fuzzy matching is robust, "Earbud_Left" might find "Earbud_Left_downsample"
        # For safety here, I use the exact name.
        try:
            signal_obj = getattr(session.signals, sig_name)
        except AttributeError:
            print(f"Skipping {sig_name}, not found.")
            continue
            
        # B. Preprocess & Analyze
        signal_obj.preprocess()
        
        results = []
        for beat in beats:
            cycle_id, start, end = int(beat[0]), int(beat[1]), int(beat[2])
            beat_res = signal_obj.analyze_beat(start, end)
            beat_res['cycle_id'] = cycle_id
            results.append(beat_res)
            
        # C. Save
        # Clean the name for the CSV filename (removing _downsample just for the file)
        clean_name = sig_name.replace("_downsample", "")
        df_res = pd.DataFrame(results)
        df_res.to_csv(f"test_data/{session.session_id}/{clean_name}_processed.csv", index=False)

# Verification
print("--- Stethoscope Results ---")
rp1_Stethoscope = pd.read_csv("test_data/1/Digital_Stethoscope_processed.csv")
print(rp1_Stethoscope.head())

print("\n--- Wrist Results ---")
rp1_Wrist = pd.read_csv("test_data/1/PPG_Wrist_processed.csv")
print(rp1_Wrist.head())
'''
