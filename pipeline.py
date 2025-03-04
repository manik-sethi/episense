import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import mne
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs
from mne.decoding import Scaler
from scipy import signal
from scipy.signal import butter, lfilter, ShortTimeFFT, get_window
from scipy.signal import ShortTimeFFT
import torch
import torch.fft

mne.set_log_level('WARNING') 
os.environ["OMP_NUM_THREADS"] = "1"  # Restrict OpenMP parallel messages
os.environ["MKL_NUM_THREADS"] = "1"  # Restrict MKL messages


class Pipeline:
    def __init__(self):
        self.raw = None
        self.fname = ""
        self.fs = 256
        self.window_size = 30
        self.overlap = 15
        self.f_low = 1
        self.f_high = 50
        self.ranges_dict = {}

    def CONFIG(self, fname, fs, window_size, overlap, f_low, f_high, ranges_dict):
        self.fname = fname
        self.fs = fs 
        self.window_size = window_size
        self.overlap = overlap
        self.f_low = f_low
        self.f_high = f_high
        self.ranges_dict = ranges_dict
    
    def read_fname(self):
        self.raw = read_raw_edf(self.fname, preload=True, verbose='ERROR')
        self.raw.filter(self.f_low, self.f_high, fir_design="firwin", skip_by_annotation="edge")
        
        # Define the gold-standard channel order (only the channels you want to keep)
        gold_standard = [
            "FP1-F7", 
            "F7-T7",
            "T7-P7",
            "P7-O1",
            "FP1-F3",
            "F3-C3",
            "C3-P3",
            "P3-O1",
            "FP2-F4",
            "F4-C4",
            "C4-P4",
            "P4-O2",
            "FP2-F8",
            "F8-T8",
            "T8-P8",   # Expected channel (we expect channels starting with T8-P8)
            "P8-O2",
            "FZ-CZ",
            "CZ-PZ",
            "P7-T7",
            "T7-FT9",
            "FT9-FT10",
            "FT10-T8"
        ]
        # Remove any duplicates from gold_standard (if they exist)
        gold_standard = list(dict.fromkeys(gold_standard))
        
        # Get current channel names from the raw data.
        current_channels = self.raw.info['ch_names']
        
        # Define a helper: return True if ch starts with any gold standard entry.
        def channel_in_gold(ch):
            for g in gold_standard:
                if ch.startswith(g):
                    return True
            return False

        # Identify channels not matching any gold standard entry.
        channels_to_drop = [ch for ch in current_channels if not channel_in_gold(ch)]
        if channels_to_drop:
            # print("Dropping channels:", channels_to_drop)
            self.raw.drop_channels(channels_to_drop)
        
        # Build new order list: For each expected gold channel, find the first matching channel in raw.
        new_order = []
        for g in gold_standard:
            for ch in self.raw.info['ch_names']:
                if ch.startswith(g):
                    if ch not in new_order:
                        new_order.append(ch)
                        break  # Only take the first matching channel.
        try:
            self.raw.reorder_channels(new_order)
        except ValueError as e:
            print("Error in reordering channels:", e)

        # Debug: print the final channel order
        # print("Final channel order:", self.raw.info['ch_names'])

    def create_epochs(self):
        epoch_list = []
        labels = []
        check_list = {}
        max_time = self.raw.times[-1]  # Last valid time in the EEG recording
        test = 0

        # Debug: Print type and content of ranges_dict
        # print("Type of ranges_dict:", type(self.ranges_dict))
        # print("Content of ranges_dict:", self.ranges_dict)

        # Iterate over each label ("Preictal", "Interictal") and its list of intervals.
        for sub_label, intervals in self.ranges_dict.items():
            for start, end in intervals:
                test += 1
                if start >= max_time:
                    # print(f"⚠️ Skipping range ({start}, {end}) - Start time exceeds max available time {max_time:.2f}s")
                    continue
                if end > max_time:
                    # print(f"⚠️ Adjusting end time from {end} to max available time {max_time:.2f}s")
                    end = max_time
                # print(f"Creating epochs from {start} to {end}, Label: {sub_label}")
                truncated_segment = self.raw.copy().crop(tmin=start, tmax=end)
                n_channels = len(truncated_segment.info['ch_names'])
                if n_channels != 22:
                    print(f"Mismatch at epoch {test}: Expected 22 channels, got {n_channels}")
                    print("Channels in this segment:", truncated_segment.info['ch_names'])
                channels, times = truncated_segment.get_data(return_times=True)
                epochs = self.apply_stft(channels)
                epoch_list.append(epochs)
                check_list[f'time{test}: {sub_label}'] = len(epochs[0][0])
                t = epochs.shape[-1]
                labels.extend([sub_label] * t)
                # print(len(epoch_list), len(labels), sub_label, check_list)
                print(f"Epoch {test} from file {os.path.basename(self.fname)} processed, label {sub_label}, {n_channels} channels")
        return epoch_list, labels

    def get_band_indices(self, frequencies, bands):
        indices = {}
        for band, (fmin, fmax) in bands.items():
            indices[band] = np.where((frequencies >= fmin) & (frequencies <= fmax))[0]
        return indices

    def apply_stft(self, channels):
        win = int(self.window_size * self.fs)
        hop = int(self.overlap * self.fs)
        spectograms_list = []
        SFT = ShortTimeFFT(win=np.hanning(win), hop=hop, fs=self.fs)
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        for channel in channels:
            S = SFT.stft(channel)
            Sxx = 10 * np.log10(S + 1e-10)
            frequencies = np.fft.rfftfreq(win, d=1/self.fs)
            band_indices = self.get_band_indices(frequencies, bands)
            band_powers = []
            for band in bands.keys():
                indices = band_indices[band]
                band_power = np.mean(Sxx[indices], axis=0)
                band_powers.append(band_power)
            channel_specs = np.stack(band_powers)  # Shape: (5, time_bins)
            spectograms_list.append(channel_specs)
        # print("---SHAPE OF ONE RANGE EPOCHING---")
        # print(np.shape(spectograms_list))
        return np.stack(spectograms_list)

    def run_pipeline(self):
        self.read_fname()
        # print("read file")
        combined_epochs, labels = self.create_epochs()
        # print("created epochs")
        return combined_epochs, labels
