import scipy
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pyampd.ampd import find_peaks
from flatline_detect import is_signal_flat_lined


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyquist
    high = highcut / nyquist

    # 设计Butterworth带通滤波器
    sos = scipy.signal.butter(order, [low, high], btype='band', output='sos')

    # 使用filtfilt进行零相位滤波
    y = scipy.signal.sosfiltfilt(sos, data)
    return y


def butter_highpass_filter(data, lowcut, fs, order=3):
    nyquist = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyquist

    # 设计Butterworth带通滤波器
    sos = scipy.signal.butter(order, low, btype='high', output='sos')

    # 使用filtfilt进行零相位滤波
    y = scipy.signal.sosfiltfilt(sos, data)
    return y

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

def signals_cleaner(input_signal, Label, target_fs, subject_num=1):
    save_path = 'cleaned_signal'
    BP_fs = 125
    input_signal = input_signal.numpy()
    ECG = input_signal[:, 0]
    PPG = input_signal[:, 1]
    if input_signal.shape[1] == 3:
        BP = input_signal[:, 2]

    all_signals_segments = []
    all_BP_segment = []
    for i in range(len(ECG)):
        BP_Label = Label[i]
        if input_signal.shape[1] == 3:
            BP_segment = BP[i]
            flat_BP = is_signal_flat_lined(BP_segment, fs=125, flat_time=0.1, signal_time=10,
                                           flat_threshold=0.25, change_threshold=0.002, moving_average=True)
            if flat_BP:
                print('BP flat')
                plt.plot(BP_segment)
                plt.savefig(save_path+f'/sub_{subject_num}_seg_{i}_BP_flat')
                plt.close()
                continue

            ## cumputation BP Peak and valleys
            peaks = find_peaks(BP_segment, scale=int(BP_fs))
            valleys = find_peaks(BP_segment.max() - BP_segment, scale=int(BP_fs))
            pk_th = 0.6
            ### Remove first or last if equal to 0 or len(sig)-1
            if peaks[0] == 0: peaks = peaks[1:]
            if valleys[0] == 0: valleys = valleys[1:]
            if peaks[-1] == len(BP_segment) - 1: peaks = peaks[:-1]
            if valleys[-1] == len(BP_segment) - 1: valleys = valleys[:-1]

            ### REMOVE THE FIRST AND LAST PEAK/VALLEY
            if peaks[0] < valleys[0]:
                peaks = peaks[1:]
            else:
                valleys = valleys[1:]

            if peaks[-1] > valleys[-1]:
                peaks = peaks[:-1]
            else:
                valleys = valleys[:-1]

            ### START AND END IN VALLEYS
            while len(peaks) != 0 and peaks[0] < valleys[0]:
                peaks = peaks[1:]

            while len(peaks) != 0 and peaks[-1] > valleys[-1]:
                peaks = peaks[:-1]

            if len(peaks) == 0 or len(valleys) == 0:
                continue

            ## Remove consecutive peaks with one considerably under the other
            new_peaks = []
            mean_vly_amp = np.mean(BP_segment[valleys])

            for peak_i in range(len(peaks) - 1):
                if BP_segment[peaks[peak_i]] - mean_vly_amp > (BP_segment[peaks[peak_i + 1]] - mean_vly_amp) * pk_th:
                    new_peaks.append(peaks[peak_i])
                    a = peak_i
                    break

            if len(peaks) == 1:
                new_peaks.append(peaks[0])
                a = 0

            for peak_j in range(a + 1, len(peaks)):
                if BP_segment[peaks[peak_j]] - mean_vly_amp > (BP_segment[new_peaks[-1]] - mean_vly_amp) * pk_th:
                    new_peaks.append(peaks[peak_j])

            new_valleys = []
            for p in peaks:
                p_vly = valleys[valleys < p][-1]
                if len(new_valleys) == 0 or new_valleys[-1] != p_vly:
                    new_valleys.append(p_vly)

            if len(new_valleys) == 0 or new_valleys[-1] != valleys[-1]:
                new_valleys.append(valleys[-1])

            ## Filter those samples with many peaks and valleys
            upper_pks = 24
            lower_pks = 1

            pks_max_check = (len(new_peaks) < upper_pks) & (len(new_valleys) < upper_pks + 1)
            pks_mix_check = (len(new_peaks) > lower_pks) & (len(new_valleys) > lower_pks + 1)

            if not (pks_max_check and pks_mix_check):
                print('BP too many peaks and valleys')
                plt.plot(BP_segment)
                plt.savefig(save_path+f'/sub_{subject_num}_seg_{i}_BP_many_PaV')
                plt.close()
                continue

            ## BPM limitation
            upper_bpm = 140
            lower_bpm = 35
            peaks_bpm = BP_fs / np.median(np.diff(new_peaks)) * 60
            valleys_bpm = BP_fs / np.median(np.diff(new_valleys)) * 60

            peaks_bpm_check = (peaks_bpm < upper_bpm) and (peaks_bpm > lower_bpm)
            valley_bpm_check = (valleys_bpm < upper_bpm) and (valleys_bpm > lower_bpm)

            each_peaks_bpm_check = sum(BP_fs / np.diff(new_peaks) * 60 < peaks_bpm / 2) == 0
            each_valleys_bpm_check = sum(BP_fs / np.diff(new_valleys) * 60 < valleys_bpm / 2) == 0

            if not (peaks_bpm_check and valley_bpm_check):
                print('BP BPM')
                plt.plot(BP_segment)
                plt.savefig(save_path+f'/sub_{subject_num}_seg_{i}_BP_BPM')
                plt.close()
                continue

            if not (each_peaks_bpm_check and each_valleys_bpm_check):
                print('BP BPM variance')
                plt.plot(BP_segment)
                plt.savefig(save_path+f'/sub_{subject_num}_seg_{i}_BP_BPM_variance')
                plt.close()
                continue

        PPG_segment = PPG[i]
        ECG_segment = ECG[i]

        ## PPG Peaks and valleys computation
        peaks_PPG = find_peaks(PPG_segment, scale=int(target_fs))
        valleys_PPG = find_peaks(PPG_segment.max() - PPG_segment, scale=int(target_fs))

        ### Remove first or last if equal to 0 or len(sig)-1
        if peaks_PPG[0] == 0: peaks_PPG = peaks_PPG[1:]
        if valleys_PPG[0] == 0: valleys_PPG = valleys_PPG[1:]
        if peaks_PPG[-1] == len(PPG_segment) - 1: peaks_PPG = peaks_PPG[:-1]
        if valleys_PPG[-1] == len(PPG_segment) - 1: valleys_PPG = valleys_PPG[:-1]

        ### REMOVE THE FIRST AND LAST PEAK/VALLEY
        if peaks_PPG[0] < valleys_PPG[0]:
            peaks_PPG = peaks_PPG[1:]
        else:
            valleys_PPG = valleys_PPG[1:]

        if peaks_PPG[-1] > valleys_PPG[-1]:
            peaks_PPG = peaks_PPG[:-1]
        else:
            valleys_PPG = valleys_PPG[:-1]

        if len(peaks_PPG) == 0 or len(valleys_PPG) == 0:
            print('PPG no PaV')
            plt.plot(PPG_segment)
            plt.savefig(save_path + f'/sub_{subject_num}_seg_{i}_PPG_no_PaV')
            plt.close()
            continue

        ## BPM limitation
        upper_bpm = 140
        lower_bpm = 35
        peaks_bpm_PPG = target_fs / np.median(np.diff(peaks_PPG)) * 60
        valleys_bpm_PPG = target_fs / np.median(np.diff(valleys_PPG)) * 60

        peaks_bpm_PPG_check = (peaks_bpm_PPG < upper_bpm) and (peaks_bpm_PPG > lower_bpm)
        valley_bpm_PPG_check = (valleys_bpm_PPG < upper_bpm) and (valleys_bpm_PPG > lower_bpm)

        each_peaks_bpm_PPG_check = sum(target_fs / np.diff(peaks_PPG) * 60 < peaks_bpm_PPG / 2) == 0
        each_valleys_bpm_PPG_check = sum(target_fs / np.diff(valleys_PPG) * 60 < valleys_bpm_PPG / 2) == 0

        if not (peaks_bpm_PPG_check and valley_bpm_PPG_check):
            print('PPG BPM')
            plt.plot(PPG_segment)
            plt.savefig(save_path + f'/sub_{subject_num}_seg_{i}_PPG_BPM')
            plt.close()
            continue

        if not (each_peaks_bpm_PPG_check and each_valleys_bpm_PPG_check):
            print('PPG BPM variance')
            plt.plot(PPG_segment)
            plt.savefig(save_path + f'/sub_{subject_num}_seg_{i}_PPG_BPM_variance')
            plt.close()
            continue

        ## std and amp limitaion
        bpm_PPG_std_th = 100
        PPG_amp_std_th = 0.5
        peaks_bpm_PPG_std = np.std(np.diff(peaks_PPG))
        valleys_bpm_PPG_std = np.std(np.diff(valleys_PPG))

        peaks_PPG_amp_std = np.std(PPG_segment[peaks_PPG])
        valleys_PPG_amp_std = np.std(PPG_segment[valleys_PPG])

        bpm_PPG_std_check = (peaks_bpm_PPG_std < bpm_PPG_std_th) and (valleys_bpm_PPG_std < bpm_PPG_std_th)
        PPG_amp_std_check = (peaks_PPG_amp_std < PPG_amp_std_th) and (valleys_PPG_amp_std < PPG_amp_std_th)

        if not (bpm_PPG_std_check and PPG_amp_std_check):
            print('PPG amp variance')
            plt.plot(PPG_segment)
            plt.savefig(save_path + f'/sub_{subject_num}_seg_{i}_PPG_amp_variance')
            plt.close()
            continue

        ## ECG Peaks and valleys computation
        try:
            peaks_ECG = find_peaks(ECG_segment, scale=int(target_fs))
            valleys_ECG = find_peaks(ECG_segment.max() - ECG_segment, scale=int(target_fs))
        except:
            ECG_segment = butter_bandpass_filter(ECG_segment, 0.5, 35, target_fs, order=2)
            peaks_ECG = find_peaks(ECG_segment, scale=int(target_fs))
            valleys_ECG = find_peaks(ECG_segment.max() - ECG_segment, scale=int(target_fs))

        ### Remove first or last if equal to 0 or len(sig)-1
        if peaks_ECG[0] == 0: peaks_ECG = peaks_ECG[1:]
        if valleys_ECG[0] == 0: valleys_ECG = valleys_ECG[1:]
        if peaks_ECG[-1] == len(ECG_segment) - 1: peaks_ECG = peaks_ECG[:-1]
        if valleys_ECG[-1] == len(ECG_segment) - 1: valleys_ECG = valleys_ECG[:-1]

        ### REMOVE THE FIRST AND LAST PEAK/VALLEY
        if peaks_ECG[0] < valleys_ECG[0]: peaks_ECG = peaks_ECG[1:]
        else: valleys_ECG = valleys_ECG[1:]

        if peaks_ECG[-1] > valleys_ECG[-1]: peaks_ECG = peaks_ECG[:-1]
        else: valleys_ECG = valleys_ECG[:-1]
        # peaks_ECG = peaks_ECG[1:-1]
        if len(peaks_ECG) == 0:
            print('ECG no PaV')
            plt.plot(ECG_segment)
            plt.savefig(save_path + f'/sub_{subject_num}_seg_{i}_ECG_no_PaV')
            plt.close()
            continue

        ## BPM limitation
        peaks_bpm_ECG = target_fs / np.median(np.diff(peaks_ECG)) * 60
        valleys_bpm_ECG = target_fs / np.median(np.diff(valleys_ECG)) * 60

        peaks_bpm_ECG_check = (peaks_bpm_ECG < upper_bpm) and (peaks_bpm_ECG > lower_bpm)
        valley_bpm_ECG_check = (valleys_bpm_ECG < upper_bpm) and (valleys_bpm_ECG > lower_bpm)

        each_peaks_bpm_ECG_check = sum(target_fs / np.diff(peaks_ECG) * 60 < peaks_bpm_ECG / 2) == 0
        each_valleys_bpm_ECG_check = sum(target_fs / np.diff(valleys_ECG) * 60 < valleys_bpm_ECG / 2) == 0

        if not (peaks_bpm_ECG_check and valley_bpm_ECG_check):
            print('ECG BPM')
            plt.plot(ECG_segment)
            plt.savefig(save_path + f'/sub_{subject_num}_seg_{i}_ECG_BPM')
            plt.close()
            continue

        if not (each_peaks_bpm_ECG_check and each_valleys_bpm_ECG_check):
            print('ECG BPM variance')
            plt.plot(ECG_segment)
            plt.savefig(save_path + f'/sub_{subject_num}_seg_{i}_ECG_BPM_variance')
            plt.close()
            continue

        ## std and amp limitaion
        bpm_ECG_std_th = 100
        ECG_amp_std_th = 1.5
        peaks_bpm_ECG_std = np.std(np.diff(peaks_ECG))
        valleys_bpm_ECG_std = np.std(np.diff(valleys_ECG))

        peaks_ECG_amp_std = np.std(ECG_segment[peaks_ECG])
        valleys_ECG_amp_std = np.std(ECG_segment[valleys_ECG])

        bpm_ECG_std_check = (peaks_bpm_ECG_std < bpm_ECG_std_th) and (valleys_bpm_ECG_std < bpm_ECG_std_th)
        ECG_amp_std_check = (peaks_ECG_amp_std < ECG_amp_std_th) and (valleys_ECG_amp_std < ECG_amp_std_th)

        if not (bpm_ECG_std_check and ECG_amp_std_check):
            print('ECG amp variance')
            plt.plot(ECG_segment)
            plt.savefig(save_path + f'/sub_{subject_num}_seg_{i}_ECG_amp_variance')
            plt.close()
            continue

        ECG_segment = torch.from_numpy(ECG_segment.copy())
        PPG_segment = torch.from_numpy(PPG_segment.copy())
        signals_segment = torch.vstack([ECG_segment, PPG_segment])
        # signals_segment = PPG_segment[np.newaxis, :]
        all_signals_segments.append(signals_segment)
        all_BP_segment.append(BP_Label)

    all_signals_segments = torch.stack(all_signals_segments, dim=0)
    all_BP_segment = torch.vstack(all_BP_segment).squeeze()
    print(f'segment num:{len(all_signals_segments)}')
    return all_signals_segments, all_BP_segment