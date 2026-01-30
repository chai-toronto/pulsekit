import numpy as np
import pandas as pd
from signal_processors import ECG, PPG, Stethoscope, Earbud, Accelerometer, RespirationBelt, BP, Hemodynamics
from feature_processors import PAT, PWA
from dataset_processors import Dataset               

# Create the Manager
test_dataset = Dataset(
    datasetName="TestDataset",
    sessionIds=["3", "3_2", "4", "5", "41", "41_2"],
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
        BP(name="Systolic_BP_downsample", sampling_rate=1000),
        BP(name="NIBP_downsample", sampling_rate=1000),
        Hemodynamics(name="Cardiac_Output_downsample", sampling_rate=1000),
        Hemodynamics(name="Stroke_Volume_downsample", sampling_rate=1000),
        Hemodynamics(name="SVR_downsample", sampling_rate=1000),
        Hemodynamics(name="Heart_Rate_downsample", sampling_rate=1000)
    ],
    data_directory="test_data"
)

print("ECG Data Example from session 3:", test_dataset.sessions._3.signals.ECG_downsample.data[:10])

print(f"Participant 3: {test_dataset.sessions._3.age} years old, {test_dataset.sessions._3.height} cm, {test_dataset.sessions._3.weight} kg, data length: {test_dataset.sessions._3.length} samples")

test_dataset.get_cardiac_cycles(reference_signal_name="ECG_downsample")

print("Windowing type:", test_dataset.windowing_type)

print("Processing signals...")
test_dataset.process_signals(signal_list=["PPG_Finger_downsample"])
fiducial_3 = pd.read_csv("test_data/3/segments.csv")
print("Percentage of original windows retained after quality check for PPG_Finger_downsample in session 3:",
      100 * fiducial_3["PPG_Finger_downsample_quality"].sum() / len(fiducial_3))


def my_special_ppg_qc(signal, preprocessed, start, end, fiducials, windowing_type):
    segment = preprocessed[start:end]
    if np.ptp(segment) < 0.1:
        return False
    return True

print("Setting new custom quality check for PPG_Finger_downsample...")
test_dataset.set_custom_quality_check("PPG_Finger_downsample", my_special_ppg_qc)
test_dataset.process_signals(signal_list=["PPG_Finger_downsample"])

fiducial_3 = pd.read_csv("test_data/3/segments.csv")
print("Percentage of new windows retained after new quality check for PPG_Finger_downsample in session 3:",
      100 * fiducial_3["PPG_Finger_downsample_quality"].sum() / len(fiducial_3))

'''
test_dataset.process_signals(signal_list=[
    "ECG_downsample",
    "PPG_Finger_downsample",
    "PPG_Ear_downsample",
    "PPG_Wrist_downsample",
    "Digital_Stethoscope_downsample",
    "Earbud_Left_downsample",
    "Earbud_Right_downsample",
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
'''