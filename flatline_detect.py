import random

import numpy as np
import scipy


def moving_average_filter(signal, window_size):
    window_size = int(window_size)

    # Ensure the window size is odd to maintain symmetry
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2

    # Pad the signal at start and end to avoid edge effects
    padded_signal = np.pad(signal, pad_width=half_window, mode='reflect')

    window = np.ones(window_size) / window_size
    filtered_signal = np.convolve(padded_signal, window, mode='valid')

    return filtered_signal

def is_signal_flat_lined(sig, fs, flat_time, signal_time, flat_threshold=0.25, change_threshold=0.01, moving_average=False):
    signal_length = fs * signal_time
    flat_segment_length = fs * flat_time
    max = sig.max()
    min = sig.min()
    sig = (sig - min)/(max - min)

    if moving_average:
        sig = moving_average_filter(sig, 10)

    flatline_segments = detect_flatline_segments(sig, change_threshold=change_threshold, min_duration=flat_segment_length)

    total_flatline_in_signal = np.sum([end - start for start, end in flatline_segments])

    if total_flatline_in_signal / signal_length > flat_threshold:
        return True
    else:
        return False

def detect_flatline_segments(sig, min_duration, change_threshold):
    """Detects flatline segments in a signal.

    Args:
        sig (ArrayLike): Signal to be analyzed (ECG or PPG).
        min_duration (float): Mimimum duration of flat segments for flatline detection.
        change_threshold (float): Threshold for change in signal amplitude.

    Returns:
        list: List of boundaries of flatline segments.
    """

    start_indices = []
    end_indices = []
    in_flatline_segment = False

    for i in range(1, len(sig)):
        change = abs(sig[i] - sig[i - 1])

        if change <= change_threshold and sig[i] != max(sig) and sig[i] != min(sig):
            if not in_flatline_segment:
                # Start of a new flatline segment
                start_indices.append(i - 1)
                in_flatline_segment = True
        else:
            if in_flatline_segment:
                # End of a flatline segment
                end_indices.append(i - 1)
                in_flatline_segment = False

    if in_flatline_segment:
        # The last segment extends until the end of the signal
        end_indices.append(len(sig) - 1)

    # Filter segments by duration
    durations = [end - start + 1 for start, end in zip(start_indices, end_indices)]
    start_indices = [start for start, duration in zip(start_indices, durations) if duration >= min_duration]
    end_indices = [end for end, duration in zip(end_indices, durations) if duration >= min_duration]

    return list(zip(start_indices, end_indices))