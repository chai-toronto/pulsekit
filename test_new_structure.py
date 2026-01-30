import numpy as np
import pandas as pd
from PulseKit import *            

# 1. Define the signals you want to include in your pipeline
# Every signal is initialized with its name and sampling rate (Hz)
test_dataset = Dataset(
    datasetName="TestDataset",
    sessionIds=["3", "3_2", "4", "5", "41", "41_2"],
    signals=[
        ECG(name="ECG_downsample"),
        PPG(name="PPG_Finger_downsample"),
        PPG(name="PPG_Ear_downsample"),
        PPG(name="PPG_Wrist_downsample"),
        Stethoscope(name="Digital_Stethoscope_downsample"),
        Earbud(name="Earbud_Left_downsample"),
        Earbud(name="Earbud_Right_downsample"),
        Accelerometer(name="AccZ_downsample"),
        RespirationBelt(name="Respiration_Belt_downsample"),
        BP(name="Diastolic_BP_downsample"),
        BP(name="Systolic_BP_downsample"),
        BP(name="NIBP_downsample"),
        Hemodynamics(name="Cardiac_Output_downsample"),
        Hemodynamics(name="Stroke_Volume_downsample"),
        Hemodynamics(name="SVR_downsample"),
        Hemodynamics(name="Heart_Rate_downsample")
    ],
    data_directory="test_data"
)

print("ECG Data Example from session 3:", test_dataset.sessions._3.signals.ECG_downsample.data[:10])

print(f"Participant 3: {test_dataset.sessions._3.age} years old, {test_dataset.sessions._3.height} cm, {test_dataset.sessions._3.weight} kg, data length: {test_dataset.sessions._3.length} samples")

test_dataset.get_cardiac_cycles(reference_signal_name="ECG_downsample")

print("Windowing type:", test_dataset.windowing_type)

print("Processing signals...")
test_dataset.process_signals(signal_list=[
    "ECG_downsample",
    "PPG_Finger_downsample",
    "PPG_Ear_downsample",
    "PPG_Wrist_downsample",
    "Digital_Stethoscope_downsample",
    "Earbud_Left_downsample",
    "Earbud_Right_downsample",
    "AccZ_downsample",
    "Respiration_Belt_downsample",
    "Diastolic_BP_downsample",
    "Systolic_BP_downsample",
    "NIBP_downsample",
    "Cardiac_Output_downsample",
    "Stroke_Volume_downsample",
    "SVR_downsample",
    "Heart_Rate_downsample"
    ])


print("Computing features...")
test_dataset.compute_features(features=[
    PWA(dataset=test_dataset, signals=["PPG_Finger_downsample", "PPG_Ear_downsample"], apply_quality = True),
    PAT(dataset=test_dataset, signals=["PPG_Finger_downsample", "Earbud_Left_downsample", "ECG_downsample"], apply_quality = True)
])