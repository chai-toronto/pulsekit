import copy
import csv
import pickle
import numpy as np
from scipy.signal import (
    butter, cheby1, filtfilt, find_peaks, sosfiltfilt
)

def cheby1_lowpass_filter(data, lowcut, fs, order=5, rp=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = cheby1(order, rp=rp, Wn=[low], btype='low')
    y = filtfilt(b, a, data)
    return y


def butter_filter(cutoff, fs, filter_direction, order=3):
    assert filter_direction == "low" or filter_direction == "high"
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff,
                         btype=filter_direction, analog=False)
    return b, a


def apply_butterworth_filter(data, cutoff, filter_direction, fs, order=5):
    b, a = butter_filter(cutoff, fs, filter_direction, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    y = sosfiltfilt(sos, data)
    return y


def faster_median_filter(series,filter_length):
    series = np.ravel(series)
    
    series_shape = series.shape[0]
    half_filter_len = int(filter_length//2)
    template = np.zeros((filter_length,series.shape[0]+filter_length))
    for i in range(0,half_filter_len):
        template[i,:] = series[0]
    for i in range(0,half_filter_len+1):
        template[-i-1,:] = series[-1]
    for i in range(0,filter_length//2):
        template[half_filter_len+i,half_filter_len+i:half_filter_len+i+series_shape] = series
        template[i,i:i+series_shape] = series
    return np.median(template[:,filter_length//2:-filter_length//2],axis = 0),template

def get_gaussian(x, mean_, standard):
    #return (1/(np.sqrt(2*np.pi*standard**2)))*np.exp(-.5*((x-mean_)/standard)**2)
    return (1/(np.sqrt(2*np.pi*standard**2)))*np.exp(-((x-mean_)**2/(2*standard**2)))

def get_gaussian_distribution(gparams,samples):
    (mean_,standard,height) = gparams
    out = [get_gaussian(i,mean_,standard)for i in range(samples)]
    out = (np.array(out)/np.max(out))*height
    return out

def get_gaussian_mixture(gaussian_params, samples):
    gaussians = []
    for g in gaussian_params:
        gaussians.append(get_gaussian_distribution(g,samples))
    
    out = np.sum(np.array(gaussians),axis=0)
    return out

def minmax_normalize(data):
    out = data
    if np.max(data)-np.min(data) != 0:
        out = (data-np.min(data))/(np.max(data)-np.min(data))
    else:
        out = data-np.max(data)
    return out

def zscore_normalize(wave, *args):
    wave = np.array(wave)
    wave = np.add(wave,-np.mean(wave))
    wave = np.multiply(wave,1/(np.std(wave)+1e-6))
    return wave

def faster_max_filter(series,filter_length):
    series = np.ravel(series)
    
    series_shape = series.shape[0]
    half_filter_len = int(filter_length//2)
    template = np.zeros((filter_length,series.shape[0]+filter_length), dtype = float)
    for i in range(0,half_filter_len):
        template[i,:] = series[0]
    for i in range(0,half_filter_len+1):
        template[-i-1,:] = series[-1]
    for i in range(0,filter_length//2):
        template[half_filter_len+i,half_filter_len+i:half_filter_len+i+series_shape] = series
        template[i,i:i+series_shape] = series
    return np.max(template[:,filter_length//2:-filter_length//2],axis = 0),template

def faster_min_filter(series,filter_length):
    series = np.ravel(series)
    
    series_shape = series.shape[0]
    half_filter_len = int(filter_length//2)
    template = np.zeros((filter_length,series.shape[0]+filter_length), dtype = float)
    for i in range(0,half_filter_len):
        template[i,:] = series[0]
    for i in range(0,half_filter_len+1):
        template[-i-1,:] = series[-1]
    for i in range(0,filter_length//2):
        template[half_filter_len+i,half_filter_len+i:half_filter_len+i+series_shape] = series
        template[i,i:i+series_shape] = series
    return np.min(template[:,filter_length//2:-filter_length//2],axis = 0),template

def faster_averaging_filter(series,filter_length, avg_func, axis = 0):
    series = np.ravel(series)
    
    series_shape = series.shape[0]
    half_filter_len = int(filter_length//2)
    template = np.zeros((filter_length,series.shape[0]+filter_length), dtype = float)
    for i in range(0,half_filter_len):
        template[i,:] = series[0]
    for i in range(0,half_filter_len+1):
        template[-i-1,:] = series[-1]
    for i in range(0,filter_length//2):
        template[half_filter_len+i,half_filter_len+i:half_filter_len+i+series_shape] = series
        template[i,i:i+series_shape] = series
        template[i,:i] = series[:i]
    return avg_func(template[:,filter_length//2:-filter_length//2],axis = axis),template


def get_next_turnaround_point(x, anchor, direction = "forward", min_dist = 20):
    second_derivative = np.diff(x)
    signs = np.sign(second_derivative).tolist()
    signs.insert(0, 0)
    signs = np.diff(signs)
    turn_rounds = np.where(signs != 0)[0]
    out = [0]
    diffs = np.add(turn_rounds,-anchor)
    if direction == "forward":
        mask = np.where(diffs>min_dist)
        turn_rounds = turn_rounds[mask]
        if len(turn_rounds) > 0:
            #print(anchor, turn_rounds, second_derivative[turn_rounds])
            out.append(turn_rounds[0])
    elif direction == "backwards":
        mask = np.where(diffs<-min_dist)
        
        turn_rounds = turn_rounds[mask]
        if len(turn_rounds) > 0:
            out.append(turn_rounds[-1])
    else:
        diffs = np.abs(diffs)
        diffs = np.put(np.where(diffs==np.min(diffs))[0],np.inf)
        mask = np.where(diffs==np.min(diffs))
        out.append(turn_rounds[mask])
    return out[-1]

def extract_fiducial_points(clip, c_peak=None):
    # for finding first peak can also find turnaround point to negative and compare
    if not c_peak:
        temp_ = np.where(np.max(clip[:500]) == clip)[0][0]
        a = temp_
    else:
        a = c_peak

    sdptg = clip
    
    b = get_next_turnaround_point(sdptg, a, direction = "forward")
    c = get_next_turnaround_point(sdptg, b, direction = "forward")
    d = get_next_turnaround_point(sdptg, c, direction = "forward")  
    e = get_next_turnaround_point(sdptg, d, direction = "forward")
    f = get_next_turnaround_point(sdptg, e, direction = "forward")
    g = get_next_turnaround_point(sdptg, f, direction = "forward")
    z = get_next_turnaround_point(sdptg, a, direction = "backwards")
    q = get_next_turnaround_point(sdptg, z, direction = "backwards")
    return (a, b, c, d, e, f, z, q, g)


def ecg_peaks(ecg_data, fs=1000):
    detection_signal = apply_butterworth_filter(ecg_data, 5, "high", fs=fs)
    detection_signal = apply_butterworth_filter(detection_signal, 15, "low", fs=fs)
    squared_signal = detection_signal ** 2
    height_req = np.mean(squared_signal)
    rough_peaks, _ = find_peaks(squared_signal, distance=400, height=height_req)
    true_peaks = []
    search_window = int(0.05 * fs) # +/- 50ms window 
    
    for p in rough_peaks:
        start = max(0, p - search_window)
        end = min(len(ecg_data), p + search_window)
        search_slice = ecg_data[start:end]
        local_max_idx = np.argmax(search_slice)
        true_peak_idx = start + local_max_idx
        true_peaks.append(true_peak_idx)
        
    return np.array(true_peaks)

def cleanup_bp_signal(bp_waveform):
    win_size = 2000 # tolerance window, if two bad samples within win_size of each other it is a bad segment
    diff_size = 4 # fourth derivative
    group_thresh = 15000
    bp_waveform = np.array(bp_waveform)
    cleaned_up_waveform = copy.deepcopy(bp_waveform)
    temp = [0 for xx in range(diff_size)]
    temp.extend(np.abs(np.diff(bp_waveform, n=diff_size))) # adds absolute value of fourth derivative to temp
    temp = np.array(temp)
    mask =  np.where((temp>45)|(bp_waveform<20),0,1) # if abs(derivate) is very large (wave super steep) OR bp less than 20, mask to 0
    group_properties_accumulator = []
    mask_idxs = np.where((temp>45)|(bp_waveform<20))[0] # returns the index positions where mask is true (e.g. 5, 6, 200)
    for idx, i in enumerate(mask_idxs):
        if len(group_properties_accumulator)==0: # start a new bad group which starts and ends at the bad point found
            start_idx = idx
            end_idx = idx
            group_properties = [mask_idxs[start_idx],mask_idxs[end_idx]]
            while start_idx-1 > 0 and np.abs(mask_idxs[start_idx]-mask_idxs[start_idx-1])<win_size: # check previous bad point in list, if its within win_size, start of group pushed back
                group_properties[0] = mask_idxs[start_idx-1]
                start_idx -= 1
            while end_idx+1 < mask_idxs.shape[0] and np.abs(mask_idxs[end_idx]-mask_idxs[end_idx+1])<win_size: # check next bad point in list, if its within win_size, end of group pushed back
                group_properties[1] = mask_idxs[end_idx+1]
                end_idx += 1
            group_properties_accumulator.append(group_properties) # append the bad group start and end idx
        elif len(group_properties_accumulator)>0 and (mask_idxs[idx] > group_properties_accumulator[-1][1]): # is the next bad one after the end of the last added group, if yes start new group
            start_idx = idx
            end_idx = idx
            group_properties = [mask_idxs[start_idx],mask_idxs[end_idx]]
            while start_idx-1 > 0 and np.abs(mask_idxs[start_idx]-mask_idxs[start_idx-1])<win_size:
                group_properties[0] = mask_idxs[start_idx-1]
                start_idx -= 1
            while end_idx+1 < mask_idxs.shape[0] and np.abs(mask_idxs[end_idx]-mask_idxs[end_idx+1])<win_size:
                group_properties[1] = mask_idxs[end_idx+1]
                end_idx += 1
            group_properties_accumulator.append(group_properties)
    
    group_properties_accumulator = np.array(group_properties_accumulator) # array of lists of start and end of "bad groups"

    groups = []
    used = []

    for gidx, group_props in enumerate(group_properties_accumulator): # iterate through each bad group found
        if gidx not in used:
            used.append(gidx) # check off groups as we put them in meta groups
            out_group_start = group_props[0]
            out_group_end = group_props[1]
            group_end = group_props[1]
            for gidx2, group_props2 in enumerate(group_properties_accumulator):
                if gidx2 >= gidx: # for all groups after the one were examining
                    next_group_start = group_props2[0]
                    if next_group_start-group_end < group_thresh: # if it occurs after the group but within the threshold, its end is the new end
                        group_end = group_props2[1]
                        out_group_end = group_props2[1]
                        used.append(gidx2)
                    else:
                        break
            groups.append([out_group_start,out_group_end]) # append mega group to groups list
                
    for group_properties in groups:
        if group_properties[0] != group_properties[1]: # widen groups by win_size on each side and linearly interpolate between these points
            adjust_start = np.max([group_properties[0]-win_size,0])
            adjust_end = np.min([group_properties[1]+win_size,cleaned_up_waveform.shape[0]-1])
            adjust_length = (adjust_end)-(adjust_start)
            adjust = np.arange(0,adjust_length)*((bp_waveform[adjust_end]-bp_waveform[adjust_start])/adjust_length)
            adjust += bp_waveform[adjust_start]
            cleaned_up_waveform[adjust_start:adjust_end] = adjust
        elif group_properties[0] == group_properties[1]: # if only one bad point, fill value from 10 points ahead
            cleaned_up_waveform[group_properties[0]] = cleaned_up_waveform[group_properties[0]+10]

        
 
    return mask, group_properties_accumulator, cleaned_up_waveform

def cleanup_bp_signal_rm(bp_waveform):
    win_size = 2000 # tolerance window, if two bad samples within win_size of each other it is a bad segment
    diff_size = 4 # fourth derivative
    group_thresh = 15000
    bp_waveform = np.array(bp_waveform)
    cleaned_up_waveform = copy.deepcopy(bp_waveform)
    temp = [0 for xx in range(diff_size)]
    temp.extend(np.abs(np.diff(bp_waveform, n=diff_size))) # adds absolute value of fourth derivative to temp
    temp = np.array(temp)
    mask =  np.where((temp>45)|(bp_waveform<20),0,1) # if abs(derivate) is very large (wave super steep) OR bp less than 20, mask to 0
    group_properties_accumulator = []
    mask_idxs = np.where((temp>45)|(bp_waveform<20))[0] # returns the index positions where mask is true (e.g. 5, 6, 200)
    for idx, i in enumerate(mask_idxs):
        if len(group_properties_accumulator)==0: # start a new bad group which starts and ends at the bad point found
            start_idx = idx
            end_idx = idx
            group_properties = [mask_idxs[start_idx],mask_idxs[end_idx]]
            while start_idx-1 > 0 and np.abs(mask_idxs[start_idx]-mask_idxs[start_idx-1])<win_size: # check previous bad point in list, if its within win_size, start of group pushed back
                group_properties[0] = mask_idxs[start_idx-1]
                start_idx -= 1
            while end_idx+1 < mask_idxs.shape[0] and np.abs(mask_idxs[end_idx]-mask_idxs[end_idx+1])<win_size: # check next bad point in list, if its within win_size, end of group pushed back
                group_properties[1] = mask_idxs[end_idx+1]
                end_idx += 1
            group_properties_accumulator.append(group_properties) # append the bad group start and end idx
        elif len(group_properties_accumulator)>0 and (mask_idxs[idx] > group_properties_accumulator[-1][1]): # is the next bad one after the end of the last added group, if yes start new group
            start_idx = idx
            end_idx = idx
            group_properties = [mask_idxs[start_idx],mask_idxs[end_idx]]
            while start_idx-1 > 0 and np.abs(mask_idxs[start_idx]-mask_idxs[start_idx-1])<win_size:
                group_properties[0] = mask_idxs[start_idx-1]
                start_idx -= 1
            while end_idx+1 < mask_idxs.shape[0] and np.abs(mask_idxs[end_idx]-mask_idxs[end_idx+1])<win_size:
                group_properties[1] = mask_idxs[end_idx+1]
                end_idx += 1
            group_properties_accumulator.append(group_properties)
    
    group_properties_accumulator = np.array(group_properties_accumulator) # array of lists of start and end of "bad groups"

    groups = []
    used = []

    for gidx, group_props in enumerate(group_properties_accumulator): # iterate through each bad group found
        if gidx not in used:
            used.append(gidx) # check off groups as we put them in meta groups
            out_group_start = group_props[0]
            out_group_end = group_props[1]
            group_end = group_props[1]
            for gidx2, group_props2 in enumerate(group_properties_accumulator):
                if gidx2 >= gidx: # for all groups after the one were examining
                    next_group_start = group_props2[0]
                    if next_group_start-group_end < group_thresh: # if it occurs after the group but within the threshold, its end is the new end
                        group_end = group_props2[1]
                        out_group_end = group_props2[1]
                        used.append(gidx2)
                    else:
                        break
            groups.append([out_group_start,out_group_end]) # append mega group to groups list
                
    for group_properties in groups:
        if group_properties[0] != group_properties[1]: # widen groups by win_size on each side and fill nans between these
            adjust_start = np.max([group_properties[0]-win_size,0])
            adjust_end = np.min([group_properties[1]+win_size,cleaned_up_waveform.shape[0]-1])
            cleaned_up_waveform[adjust_start:adjust_end] = np.nan
        elif group_properties[0] == group_properties[1]: # if only one bad point, fill nan for just that one
            cleaned_up_waveform[group_properties[0]] = np.nan

    return mask, group_properties_accumulator, cleaned_up_waveform

def get_data_from_raw_data(tags_to_load, data_tags_to_load, root_data_folder, timestamps = True, participant_dirs = False):
    flat_data = {}
    participants = []
    for t in tags_to_load:
        if participant_dirs:
            load_dir = f"{root_data_folder}/{t}/"
        else:
            load_dir = f"{root_data_folder}/{t}"
        if True:
            for f in data_tags_to_load:
                if f not in flat_data:
                    flat_data[f] = []
                #temp = np.load(f"{load_dir}_{f}.npy", allow_pickle=True, fix_imports=True)
                temp = np.fromfile(f"{load_dir}{f}", dtype = float)
                flat_data[f].extend(temp.tolist())
            participants.extend([t for i in range(temp.shape[0])])
    flat_data["Participants_"] = participants
    return flat_data