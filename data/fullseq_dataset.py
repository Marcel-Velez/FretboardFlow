import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import pickle as pkl

from .dataset import FretboardFlow

OCTAVE = 12

class FretboardFlowFullSeq(FretboardFlow):
    """
    Inherits from FretboardFlow but, for each (song_id, version),
    cuts the entire chord progression (and MIDI frames) into all
    slices of length `history_length + 1` across the timeline,
    instead of returning a single random timestep from each file.
    """

    def __init__(self, root_dir, history_length=3, fs=100, cache_dir='./data'):
        super().__init__(root_dir, history_length=history_length, fs=fs)

        # We'll store a list of all possible samples from *all* songs/versions.
        # Each sample is the same structure as the original __getitem__ returned,
        # but we produce one sample per valid time t.
        self.fullseq_samples = []

        self.processed_data_path = os.path.join(cache_dir, f"processed_ff_full_data_hlen_{history_length}.pkl")

        # Check if processed data already exists
        if os.path.exists(self.processed_data_path):
            print("Loading preprocessed dataset...")
            with open(self.processed_data_path, "rb") as f:
                self.fullseq_samples = pkl.load(f)
        else:
            print("Processing dataset from scratch...")
            self._process_dataset()
            # Save for future use
            with open(self.processed_data_path, "wb") as f:
                pkl.dump(self.fullseq_samples, f)

    def _process_dataset(self):

        for (song_id, version) in self.song_versions:
            # 1) Load chord labels
            lab_path = os.path.join(self.lab_dir, f"{song_id}_full_done.lab")
            with open(lab_path, "r") as f:
                chord_labels = [line.strip().split("\t")[-1]
                                for line in f.readlines() if line.strip() != '']

            encoded_chords = np.array([self.encode_chord(ch) for ch in chord_labels])

            # 2) Load MIDI/fret data for this song/version
            midi_files = sorted([
                os.path.join(self.midi_dir, f) for f in os.listdir(self.midi_dir)
                if f.startswith(f"{song_id}_{version}") and f.endswith(".mid")
            ])
            midi_data, uncertainty_mask = self.process_midi(midi_files)
            # midi_data shape: (6, T, ???)
            # uncertainty_mask shape: (T, 6)

            # The total number of frames is:
            T = midi_data.shape[1]

            # 3) For each valid time t in [history_length, T-1],
            #    build a sample. This ensures we get slices for *every* t
            #    across the progression, not just a random one.
            for t in range(self.history_length, T - 1):
                # Extract the portion for this sample
                sample = self._build_sample(
                    encoded_chords, chord_labels,
                    midi_data, uncertainty_mask,
                    t, song_id, version
                )
                if sample is not None:
                    self.fullseq_samples.append(sample)

        print(f"✅ Full-sequence dataset built: total samples = {len(self.fullseq_samples)}")


    def _build_sample(self,
                      encoded_chords, textual_chords,
                      midi_data, uncertainty_data,
                      t, song_id, version):
        """
        Just like the original __getitem__ logic, but for a fixed t.
        Returns (chords_input, frets_prev_input, frets_target,
                 uncertainty_mask, song_id, version, textual_chords).
        """
        # Quick shape checks:
        T = encoded_chords.shape[0]
        if t >= T:
            return None  # out of bounds

        # uncertainty_mask for time t => shape: (6,)
        unc_mask_t = uncertainty_data[t]

        # chords_input from [t-history_length..t]
        # shape => (history_length+1, 36) if your encode_chord is 36-dim
        chords_slice = encoded_chords[t - self.history_length : t + 1]
        if chords_slice.shape[0] != (self.history_length + 1):
            return None

        # frets_prev_input => shape (6, history_length, ???)
        # i.e. for each of the 6 strings, the last `history_length` frames
        frets_prev_slice = midi_data[:6, t - self.history_length : t, :]
        # That yields shape (6, history_length, ???)
        if frets_prev_slice.shape[1] != self.history_length:
            return None

        # frets_target => the shape (6, ???) or (6,) for time t if you want
        #  "predict chord at time t" from the history.
        frets_target_slice = midi_data[:6, t]  # shape: (6, ???) or (6,)

        # Build final sample
        sample = (
            torch.tensor(chords_slice,         dtype=torch.float32),
            torch.tensor(frets_prev_slice,     dtype=torch.float32),
            torch.tensor(frets_target_slice,   dtype=torch.float32),
            torch.tensor(unc_mask_t,           dtype=torch.float32),
        )
        return sample

    def __len__(self):
        return len(self.fullseq_samples)

    def __getitem__(self, idx):
        return self.fullseq_samples[idx]
