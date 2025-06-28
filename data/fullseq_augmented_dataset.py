import os
import numpy as np
import torch
import re
import random
from torch.utils.data import Dataset
from .fullseq_dataset import FretboardFlowFullSeq
import pickle as pkl

OCTAVE = 12

class FretboardFlowFullAugmented(FretboardFlowFullSeq):
    """
    Inherits from FretboardFlow but adds chord/fret transposition
    (downwards until we hit an open string, and upwards until the 15th fret).
    """

    def __init__(self, root_dir, history_length=3, fs=100, cache_dir='./data'):
        super().__init__(root_dir, history_length=history_length, fs=fs,
                         )

        # Instead of returning a single random 't' from each song version,
        # we build a large list of (chords_input, frets_prev_input, frets_target,
        #  uncertainty_mask, song_id, version, textual_chords)
        # for all timesteps in all songs, plus their augmented versions.

        self.processed_data_path = os.path.join(cache_dir, f"processed_ff_full_aug_data_hlen_{history_length}.pkl")

        # Check if processed data already exists
        if os.path.exists(self.processed_data_path):
            print("Loading preprocessed dataset...")
            with open(self.processed_data_path, "rb") as f:
                self.augmented_samples = pkl.load(f)
        else:
            print("Processing dataset from scratch...")
            self._process_dataset()
            # Save for future use
            with open(self.processed_data_path, "wb") as f:
                pkl.dump(self.augmented_samples, f)

    def _process_dataset(self):
        self.augmented_samples = []

        # Build the entire dataset of non-augmented + augmented samples.
        for (song_id, version) in self.song_versions:
            # -- load chord labels --
            lab_path = os.path.join(self.lab_dir, f"{song_id}_full_done.lab")
            with open(lab_path, "r") as f:
                chord_labels = [line.strip().split("\t")[-1]
                                for line in f.readlines() if line.strip() != '']

            encoded_chords = np.array([self.encode_chord(ch) for ch in chord_labels])

            # -- load corresponding MIDI data (fret diagrams) --
            midi_files = sorted([
                os.path.join(self.midi_dir, f) for f in os.listdir(self.midi_dir)
                if f.startswith(f"{song_id}_{version}") and f.endswith(".mid")
            ])
            midi_data, uncertainty_data = self.process_midi(midi_files)  # shape: (6, T, 23?), (T, 6)

            # For each possible timestep t, build a base sample + augment it
            T = midi_data.shape[1]  # total frames
            for t in range(self.history_length, T - 1):
                sample = self._build_single_sample(encoded_chords, chord_labels,
                                                   midi_data, uncertainty_data,
                                                   t, song_id, version)
                # sample is a tuple:
                # (chords_input, frets_prev_input, frets_target, uncertainty_mask,
                #  song_id, version, chord_labels)

                if sample is not None:
                    # Add the original sample
                    self.augmented_samples.append(sample)

                    # Generate downward shifts
                    self._generate_downward_shifts(sample)

                    # Generate upward shifts
                    self._generate_upward_shifts(sample)

        print(f"✅ Augmented dataset: total samples = {len(self.augmented_samples)}")


    def _build_single_sample(self, encoded_chords, textual_chords,
                             midi_data, uncertainty_data, t,
                             song_id, version):
        """
        Creates one (non-augmented) training sample at time t.
        Returns the tuple
          (chords_input, frets_prev_input, frets_target, uncertainty_mask,
           song_id, version, textual_chords)
        or None if there's an issue.
        """
        # Quick check on array shapes
        if t >= midi_data.shape[1]:
            return None

        # Extract uncertainty mask for time t
        # shape: (6,)
        uncertainty_mask = uncertainty_data[t]

        # chord_input: from t-history_length to t (inclusive)
        chords_input = encoded_chords[t - self.history_length : t + 1]  # shape: (history_length+1, 36)
        if chords_input.shape[0] != (self.history_length + 1):
            return None

        # frets_prev_input: shape (6, history_length, 23?)
        frets_prev_input = midi_data[:6, t - self.history_length : t, :]  # 6 x history_length x 23
        if frets_prev_input.shape[1] != self.history_length:
            return None

        # frets_target: shape (6,) at time t
        # This might be a one-hot or integer representation depending on your code
        frets_target = midi_data[:6, t]  # => shape (6, 23?) if one-hot or (6,) if integer-based

        sample = (
            torch.tensor(chords_input, dtype=torch.float32),
            torch.tensor(frets_prev_input, dtype=torch.float32),
            torch.tensor(frets_target,    dtype=torch.float32),
            torch.tensor(uncertainty_mask, dtype=torch.float32),
        )
        return sample


    # -------------------------------------------------------------------------
    #  AUGMENTATION LOGIC
    # -------------------------------------------------------------------------

    def _generate_downward_shifts(self, base_sample):
        """
        Shift the chord + fret diagrams down semitone by semitone
        until we hit a chord that has an open string (fret=0 in the diagram).
        Each valid shift is appended to self.augmented_samples.
        """
        shifted = base_sample
        while True:
            shifted = self._shift_sample(shifted, shift=-1)
            if shifted is None:
                break
            # If the new chord or target had an open string => stop.
            # But if you're *allowed* to keep going until you *cause* an open string,
            # you'd check inside `_shift_sample` or here.
            # The standard approach from [12] is "shift down until an open string appears."
            # So if the *resulting* sample has an open string, we STOP *before* adding it.
            if self._has_open_string(shifted[1], shifted[2]):
                # We stop but do NOT add the sample that triggered open strings
                break
            # Otherwise it's valid => add it
            self.augmented_samples.append(shifted)

    def _generate_upward_shifts(self, base_sample):
        """
        Shift the chord + fret diagrams up semitone by semitone
        until we exceed fret 15.
        """
        shifted = base_sample
        while True:
            shifted = self._shift_sample(shifted, shift=+1)
            if shifted is None:
                break

            # If we exceed fret=15, we stop (and do NOT add).
            if self._exceeds_fret_15(shifted[1], shifted[2]):
                break

            # Otherwise it's valid => add
            self.augmented_samples.append(shifted)


    def _shift_sample(self, sample, shift=-1):
        """
        Given a sample (chords_input, frets_prev_input, frets_target, ...),
        produce a new sample transposed by `shift` semitones:
          - chords_input => shift the root & bass
          - frets_prev_input => shift the entire fret diagram
          - frets_target => shift the entire fret diagram
        Return None if the shift is impossible or if something breaks.
        """
        (chords_input, frets_prev_input, frets_target, uncertainty_mask) = sample

        # 1) SHIFT THE CHORDS by shift semitones
        #    chords_input is shape: (history_length+1, 36)
        #    but that 36 is (12 root + 12 pitches + 12 bass) = 3*OCTAVE
        #    We'll shift the "root" part + the "bass" part by `shift`.
        #    We'll also shift "pitches" internally, but that’s trickier
        #    because it's an actual chord shape. The simplest approach is
        #    to decode back to chord label, shift semitone, re-encode.
        #    Or do a direct circular shift of the 12-dim pitch vector.
        #    For demonstration, let's do a circular shift in the root/pitch/bass vectors.

        # (A) Convert to numpy for easier slicing
        chords_input_np = chords_input.numpy().copy()
        #    For each row (time), we do root, pitch, bass
        for i in range(chords_input_np.shape[0]):
            row = chords_input_np[i]
            root_vec = row[0:12]      # one-hot
            pitch_vec = row[12:24]    # one-hot
            bass_vec = row[24:36]     # one-hot

            # find which index is 1 in root_vec
            old_root = root_vec.argmax()
            new_root = (old_root + shift) % 12
            root_vec[:] = 0
            root_vec[new_root] = 1

            # shift pitch_vec by `shift` semitones circularly
            pitch_vec[:] = np.roll(pitch_vec, shift)

            # shift bass
            old_bass = bass_vec.argmax()
            new_bass = (old_bass + shift) % 12
            bass_vec[:] = 0
            bass_vec[new_bass] = 1

        # 2) SHIFT THE FRETS in frets_prev_input & frets_target
        #    frets_prev_input => shape (6, history_length, 23?) if one-hot,
        #    or shape (6, history_length) if integer-based.
        #    We’ll assume it’s *one-hot per fret* for demonstration.
        #    Then shifting by 1 semitone means shifting that one-hot
        #    along the fret dimension by +1 or -1 index – *unless* we are
        #    at 0 or 22 or so. If integer-based, it's simpler: new_fret = old_fret + shift.

        frets_prev_np = frets_prev_input.numpy().copy()
        frets_target_np = frets_target.numpy().copy()

        # Example: if shape is (6, history_length, 23), we do a shift for each string/time
        # on the last dimension. We'll do a naive approach: argmax->shift->clip->re-onehot
        if frets_prev_np.ndim == 3:
            # SHIFT frets_prev
            for string_idx in range(frets_prev_np.shape[0]):
                for hist_t in range(frets_prev_np.shape[1]):
                    old_fret = frets_prev_np[string_idx, hist_t].argmax()
                    new_fret = old_fret + shift
                    if new_fret < 0 or new_fret > 22:
                        return None  # out of range => invalid
                    # re-onehot
                    new_vec = np.zeros_like(frets_prev_np[string_idx, hist_t])
                    new_vec[new_fret] = 1
                    frets_prev_np[string_idx, hist_t] = new_vec

            # SHIFT frets_target
            for string_idx in range(frets_target_np.shape[0]):
                old_fret = frets_target_np[string_idx].argmax()
                new_fret = old_fret + shift
                if new_fret < 0 or new_fret > 22:
                    return None
                new_vec = np.zeros_like(frets_target_np[string_idx])
                new_vec[new_fret] = 1
                frets_target_np[string_idx] = new_vec

        else:
            # Possibly integer-based shape: (6, history_length) or (6,)
            # We'll assume it's (6, history_length) for frets_prev and (6,) for frets_target
            if frets_prev_np.ndim == 3:
                return None  # handle or raise an error
            for s in range(frets_prev_np.shape[0]):
                for hist_t in range(frets_prev_np.shape[1]):
                    old_fret = frets_prev_np[s, hist_t]
                    new_fret = old_fret + shift
                    # If old_fret < 0 => muted => keep it muted (?)
                    # Or do you shift a muted string? Typically you wouldn't un-mute it
                    # if it's actually muted. There's a design decision here. We'll
                    # do nothing for muted or open strings.
                    # That’s up to your interpretation of [12].
                    if old_fret >= 0:
                        if new_fret < 0 or new_fret > 22:
                            return None
                        frets_prev_np[s, hist_t] = new_fret

            # SHIFT frets_target
            for s in range(frets_target_np.shape[0]):
                old_fret = frets_target_np[s]
                if old_fret >= 0:
                    new_fret = old_fret + shift
                    if new_fret < 0 or new_fret > 22:
                        return None
                    frets_target_np[s] = new_fret

        # Rebuild the new sample
        chords_input_shifted    = torch.tensor(chords_input_np,   dtype=torch.float32)
        frets_prev_input_shifted= torch.tensor(frets_prev_np,     dtype=torch.float32)
        frets_target_shifted    = torch.tensor(frets_target_np,   dtype=torch.float32)

        new_sample = (
            chords_input_shifted,
            frets_prev_input_shifted,
            frets_target_shifted,
            uncertainty_mask,  # unchanged
        )
        return new_sample


    def _has_open_string(self, frets_prev_input, frets_target):
        """
        Check if the *shifted* chord diagram includes an open string
        (which we interpret as fret=0).
        If either the 'frets_prev_input' or the 'frets_target'
        has an open string, we return True.
        """
        # For one-hot shape: (6, hist_len, 23) => we can check argmax.
        # For integer shape: (6, hist_len) => we check direct values.
        prev_np = frets_prev_input.numpy()
        targ_np = frets_target.numpy()

        if prev_np.ndim == 3:
            # shape (6, hist_len, 23)? => check argmax
            # If any string/time has argmax==0 => open string
            if np.any(prev_np.argmax(axis=-1) == 0):
                return True
        else:
            # shape (6, hist_len)? => check if == 0
            if np.any(prev_np == 0):
                return True

        # now target
        if targ_np.ndim == 2:  # shape (6, 23)? => check argmax
            if np.any(targ_np.argmax(axis=-1) == 0):
                return True
        else:
            # shape (6,) => check if == 0
            if np.any(targ_np == 0):
                return True

        return False

    def _exceeds_fret_15(self, frets_prev_input, frets_target):
        """
        Check if we have any fret > 15 in either the 'frets_prev_input' or 'frets_target'.
        """
        prev_np = frets_prev_input.numpy()
        targ_np = frets_target.numpy()

        if prev_np.ndim == 3:
            # shape (6, hist_len, 23) one-hot => fret = argmax
            if (prev_np.argmax(axis=-1) > 15).any():
                return True
        else:
            # integer shape
            if (prev_np > 15).any():
                return True

        # check target
        if targ_np.ndim == 2:  # (6, 23) => one-hot => check argmax
            if (targ_np.argmax(axis=-1) > 15).any():
                return True
        else:
            # integer
            if (targ_np > 15).any():
                return True

        return False


    # -------------------------------------------------------------------------
    #  OVERRIDE __len__ and __getitem__ to use self.augmented_samples
    # -------------------------------------------------------------------------

    def __len__(self):
        return len(self.augmented_samples)

    def __getitem__(self, idx):
        # print(self.augmented_samples[idx][0].shape, self.augmented_samples[idx][1].shape, \
        #       self.augmented_samples[idx][2].shape, self.augmented_samples[idx][3].shape, \
        #       len(self.augmented_samples[idx][4]),  len(self.augmented_samples[idx][5]),
        #       )
        return self.augmented_samples[idx]
