import sys
import os
import pandas as pd
import numpy as np
from signal_processors import ECG, PPG, Stethoscope, Earbud, Accelerometer, RespirationBelt
from dataset_processors import Dataset               

# Create the Manager
test_dataset = Dataset(
    datasetName="TestDataset",
    sessionIds=["1", "2", "2_2", "3", "3_2", "4", "5"],
    signals=[
        ECG(name="ECG_downsample", sampling_rate=1000),
        PPG(name="PPG_Finger_downsample", sampling_rate=1000),
        PPG(name="PPG_Ear_downsample", sampling_rate=1000),
        PPG(name="PPG_Wrist_downsample", sampling_rate=1000),
        Stethoscope(name="Digital_Stethoscope_downsample", sampling_rate=1000),
        Earbud(name="Earbud_Left_downsample", sampling_rate=1000),
        Earbud(name="Earbud_Right_downsample", sampling_rate=1000),
        Accelerometer(name="AccZ_downsample", sampling_rate=1000),
        RespirationBelt(name="Respiration_Belt_downsample", sampling_rate=1000)
    ],
    data_directory="test_data"
)

print(f"Session 1 Loaded. ECG Shape: {test_dataset.sessions._1.signals.ECG_downsample.data.shape}")
print(f"Session 3 Loaded. PPG Finger Shape: {test_dataset.sessions._3.signals.PPG_Finger_downsample.data.shape}")

#test_dataset.get_cardiac_cycles(reference_signal_name="ECG_downsample")
test_dataset.get_fixed_windows(length=15)

print(f"Session 1 Signal Windows: {test_dataset.sessions._1.windows.shape}")
print(f"Session 3 Signal Windows: {test_dataset.sessions._3.windows[:5, :]}")

test_dataset.process_signals(signal_list=[
    "PPG_Finger_downsample",
    "PPG_Ear_downsample",
    "PPG_Wrist_downsample",
    "Digital_Stethoscope_downsample",
    "Earbud_Left_downsample",
    "Earbud_Right_downsample"
    ])
