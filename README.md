# pulsekit

About: PulseKit is a data processing library for use alongside the OUR DATASET NAME (LINK).
It enables a data pipleline to be built, which the user can select which signals it wishes to include.
The dataset object, when initialized, will create Session objects for each data session, and within each of those a specific instance of each PhysiologicalSignal object which you have included in your pipeline (one of PPG, ECG, DigitalStethscope, etc.).
The session objects will store information about the user, such as height, weight, age, gender, and participant_id.
The PhysiologicalSignal subclasses (PhysiologicalSignal, ECG, PPG, Earbud, Stethoscope, Accelerometer, RespirationBelt, BP, SVR, CO, HR) contain specificc signal processing code for the preprocessing, fiducial point extraction (if applicable), and quality checks that are applied across all sessions for these signals.

## Initialization

