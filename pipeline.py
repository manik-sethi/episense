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
        self.raw.filter(l_freq=1.0, h_freq=None, fir_design="firwin", skip_by_annotation="edge")

        self.raw.notch_filter(freqs=[60, 120], notch_widths=[2.0, 2.0])
        
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

    def create_epochs(self, raw_channels, ranges_dict):
        """
        raw_channels: ndarray, shape (n_channels, n_samples)
        ranges_dict: {'Preictal': [(start, end), ...], 'Interictal': [(start, end), ...]}
                    times in seconds
        """
        fs         = self.fs
        win_sec    = self.window_size      # e.g. 30
        hop_sec    = win_sec * 0.25        # 25% stride → 75% overlap
        win_samp   = int(win_sec * fs)
        hop_samp   = int(hop_sec * fs)

        pre_epochs = []
        int_epochs = []

        # 1) Preictal: sliding windows
        for (t0, t1) in ranges_dict['Preictal']:
            start_i = int(t0 * fs)
            end_i   = int(t1 * fs)
            i       = start_i
            while i + win_samp <= end_i:
                segment = raw_channels[:, i : i + win_samp]  # shape (n_ch, win_samp)
                pre_epochs.append(segment)
                i += hop_samp

        # 2) Interictal: one window per interval
        for (t0, t1) in ranges_dict['Interictal']:
            start_i = int(t0 * fs)
            end_i   = int(t1 * fs)
            if end_i - start_i >= win_samp:
                # if longer than window, you might choose to slide here too,
                # but paper uses one per interictal, so:
                segment = raw_channels[:, start_i : start_i + win_samp]
            else:
                # pad or skip if too short
                pad_width = win_samp - (end_i - start_i)
                seg       = raw_channels[:, start_i:end_i]
                segment   = np.pad(seg, ((0,0),(0,pad_width)), mode='constant', constant_values=0)
            int_epochs.append(segment)

        # 3) Balance classes by downsampling the larger set
        n_pre = len(pre_epochs)
        n_int = len(int_epochs)

        if n_pre > n_int:
            pre_epochs = random.sample(pre_epochs, n_int)
        elif n_int > n_pre:
            int_epochs = random.sample(int_epochs, n_pre)
        # now n_pre == n_int

        # 4) Stack and label
        X = np.stack(pre_epochs + int_epochs)  # shape: (2*N, n_ch, win_samp)
        y = np.array([1]*len(pre_epochs) + [0]*len(int_epochs))

        return X, y

    def get_band_indices(self, frequencies, bands):
        indices = {}
        for band, (fmin, fmax) in bands.items():
            indices[band] = np.where((frequencies >= fmin) & (frequencies <= fmax))[0]
        return indices

    def apply_stft(self, channels):
        win = int(self.window_size * self.fs)
        hop = int(self.overlap * self.fs)
        spectograms = []
        SFT = ShortTimeFFT(win=np.hanning(win), hop=hop, fs=self.fs)

        for channel in channels:
            S = SFT.stft(channel)
            Sxx = 10 * np.log10(S + 1e-10)
            
            spectograms.append(Sxx)
        # resulting shape: (n_channels, freq_bins, time_bins)
        return np.stack(spectograms)

    def run_pipeline(self):
        # 1) load & filter EDF
        self.read_fname()

        # 2) pull out the raw numpy array: shape (n_channels, n_samples)
        raw_data = self.raw.get_data()

        # 3) epoch into windows + balance classes
        X_epochs, y = self.create_epochs(raw_data, self.ranges_dict)
        #    X_epochs.shape == (n_epochs, n_ch, win_samp)

        # 4) apply STFT to each epoch
        specs = [ self.apply_stft(epoch) for epoch in X_epochs ]
        #    each spec.shape == (n_ch, n_freq_bins, n_time_bins)

        specs = np.stack(specs)  
        #    specs.shape == (n_epochs, n_ch, n_freq_bins, n_time_bins)

        return specs, y

