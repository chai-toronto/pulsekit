'''
Helper functions for signal processing, including filtering, peak detection, and fiducial point extraction.
'''

import copy
import numpy as np
from scipy.signal import (
    butter, filtfilt
)


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