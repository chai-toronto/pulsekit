import sys
import os
import pandas as pd
import numpy as np
from signal_processors import ECG, PPG, Stethoscope, Earbud, Accelerometer, RespirationBelt, BP
from feature_processors import Feature, PAT, PWA
from dataset_processors import Dataset               

# Create the Manager
test_dataset = Dataset(
    datasetName="TestDataset",
    sessionIds=["3", "3_2", "4", "5"],
    signals=[
        ECG(name="ECG_downsample", sampling_rate=1000),
        PPG(name="PPG_Finger_downsample", sampling_rate=1000),
        PPG(name="PPG_Ear_downsample", sampling_rate=1000),
        PPG(name="PPG_Wrist_downsample", sampling_rate=1000),
        Stethoscope(name="Digital_Stethoscope_downsample", sampling_rate=1000),
        Earbud(name="Earbud_Left_downsample", sampling_rate=1000),
        Earbud(name="Earbud_Right_downsample", sampling_rate=1000),
        Accelerometer(name="AccZ_downsample", sampling_rate=1000),
        RespirationBelt(name="Respiration_Belt_downsample", sampling_rate=1000),
        BP(name="Diastolic_BP_downsample", sampling_rate=1000),
        BP(name="Systolic_BP_downsample", sampling_rate=1000)
    ],
    data_directory="test_data"
)

print(f"Participant 3: {test_dataset.sessions._3.age} years old, {test_dataset.sessions._3.height} cm, {test_dataset.sessions._3.weight} kg")
      
test_dataset.get_cardiac_cycles(reference_signal_name="ECG_downsample")

print("Windowing type:", test_dataset.windowing_type)

test_dataset.process_signals(signal_list=[
    "ECG_downsample",
    "PPG_Finger_downsample",
    "PPG_Ear_downsample",
    "PPG_Wrist_downsample",
    "Digital_Stethoscope_downsample",
    "Earbud_Left_downsample",
    "Earbud_Right_downsample",
    "Diastolic_BP_downsample",
    "Systolic_BP_downsample"
    ])


print("Computing features...")
test_dataset.compute_features(features=[
    PWA(dataset=test_dataset, signals=["PPG_Finger_downsample", "PPG_Ear_downsample"]),
    PAT(dataset=test_dataset, signals=["PPG_Finger_downsample", "Earbud_Left_downsample", "ECG_downsample"])
])
