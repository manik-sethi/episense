import glob, os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import mne
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs
from mne.decoding import Scaler
from scipy.signal import ShortTimeFFT, stft
import torch
import torch.fft

mne.set_log_level('WARNING') 
os.environ["OMP_NUM_THREADS"] = "1"  # Restrict OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # Restrict MKL


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
        self.raw.filter(
            l_freq=1.0,
            h_freq=None,
            fir_design="firwin",
            skip_by_annotation="edge",
            n_jobs=4
        )
        self.raw.notch_filter(
            freqs=[60.0, 120.0],
            notch_widths=2.0,
            n_jobs=4
        )
        
        # Gold-standard channel list
        gold_standard = [
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
            "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9",
            "FT9-FT10", "FT10-T8"
        ]
        gold_standard = list(dict.fromkeys(gold_standard))

        current = self.raw.info['ch_names']
        def channel_in_gold(ch):
            return any(ch.startswith(g) for g in gold_standard)
        to_drop = [ch for ch in current if not channel_in_gold(ch)]
        if to_drop:
            self.raw.drop_channels(to_drop)

        new_order = []
        for g in gold_standard:
            for ch in self.raw.info['ch_names']:
                if ch.startswith(g) and ch not in new_order:
                    new_order.append(ch)
                    break
        try:
            self.raw.reorder_channels(new_order)
        except ValueError as e:
            print("Error reordering channels:", e)

    def create_epochs(self, raw_channels, ranges_dict):
        fs       = self.fs
        win_sec  = self.window_size
        hop_sec  = win_sec * 0.25
        win_samp = int(win_sec * fs)
        hop_samp = int(hop_sec * fs)

        pre_epochs = []
        int_epochs = []
        
        # Preictal sliding windows
        for (t0, t1) in ranges_dict.get('Preictal', []):
            start = int(t0 * fs)
            end   = int(t1 * fs)
            i = start
            while i + win_samp <= end:
                pre_epochs.append(raw_channels[:, i:i+win_samp])
                i += hop_samp

        # Interictal one window
        for (t0, t1) in ranges_dict.get('Interictal', []):
            start = int(t0 * fs)
            end   = int(t1 * fs)
            length = end - start
            if length >= win_samp:
                segment = raw_channels[:, start:start+win_samp]
            else:
                seg = raw_channels[:, start:end]
                pad = win_samp - seg.shape[1]
                segment = np.pad(seg, ((0,0),(0,pad)), 'constant')
            int_epochs.append(segment)

        # Balance only if both classes exist
        n_pre = len(pre_epochs)
        n_int = len(int_epochs)
        if n_pre > 0 and n_int > 0:
            if n_pre > n_int:
                pre_epochs = random.sample(pre_epochs, n_int)
            elif n_int > n_pre:
                int_epochs = random.sample(int_epochs, n_pre)

        all_epochs = pre_epochs + int_epochs
        if not all_epochs:
            return np.empty((0, raw_channels.shape[0], win_samp)), np.empty((0,), dtype=int)

        X = np.stack(all_epochs)
        y = np.array([1]*len(pre_epochs) + [0]*len(int_epochs), dtype=int)
        return X, y

    def apply_stft(self, channels):
        # 30 s window → 7680 samples; no overlap inside the window → one time bin per epoch
        win       = int(self.window_size * self.fs)   # 7680
        noverlap  = 0                                # so that each epoch gives exactly one STFT column
        nfft      = 256                              # your desired FFT length
        spec_list = []

        for ch in channels:
            # Compute STFT on the entire 30 s epoch:
            f, t, Zxx = stft(
                ch,
                fs=self.fs,
                window='hann',
                nperseg=win,
                noverlap=noverlap,
                nfft=win,
                boundary=None,
                padded=False
            )
            # Zxx.shape == (nfft//2 + 1, 1)
            Sxx = 10 * np.log10(np.abs(Zxx) + 1e-10)

            # Restrict to 1–50 Hz if desired:
            freq_mask = (f >= self.f_low) & (f <= self.f_high)
            Sxx = Sxx[freq_mask, :]

            spec_list.append(Sxx)

        # Returns array with shape (n_channels, n_selected_bins, 1)
        return np.stack(spec_list)

    def run_pipeline(self):
        # Load & filter
        self.read_fname()
        raw = self.raw.get_data()

        # Epoch
        X_epochs, y = self.create_epochs(raw, self.ranges_dict)
        if X_epochs.size == 0:
            return np.empty((0, raw.shape[0], 0, 0)), y

        # STFT
        specs = []
        for epoch in X_epochs:
            specs.append(self.apply_stft(epoch))
        if not specs:
            return np.empty((0, raw.shape[0], 0, 0)), y

        specs = np.stack(specs)
        return specs, y
