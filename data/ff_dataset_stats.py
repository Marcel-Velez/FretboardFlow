import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import pickle as pkl

import pandas as pd
import re

OCTAVE = 12

class FretboardFlow(Dataset):
    def __init__(self, root_dir, history_length=3, fs=100):
        """
        Args:
            root_dir (str): Parent directory containing "midi" and "lab_files" subdirectories.
            fs (int): Frame rate for MIDI conversion.
        """
        self.root_dir = root_dir
        self.midi_dir = os.path.join(root_dir, "midi")
        self.lab_dir = os.path.join(root_dir, "lab_files")
        self.fs = fs
        self.history_length = history_length

        # Load all song IDs
        self.song_ids = sorted(set(f.split("_")[0] for f in os.listdir(self.lab_dir) if f.endswith(".lab")))

        # Build list of (song_id, version) pairs
        self.song_versions = []
        unique_song_ids = set(f.split("_")[0] for f in os.listdir(self.lab_dir) if f.endswith(".lab"))

        for song_id in unique_song_ids:

            versions = sorted(set(f.split("_")[1] for f in os.listdir(self.midi_dir) if f.startswith(song_id+'_')))
            for version in versions:
                midi_files = sorted([
                    f for f in os.listdir(self.midi_dir)
                    if f.startswith(f"{song_id}_{version}") and f.endswith(".mid")
                ])

                if len(midi_files) == 6:  # Only add versions with exactly 6 MIDI files
                    self.song_versions.append((song_id, version))
                else:
                    print(f"⚠️ Warning: {song_id}_{version} has {len(midi_files)} MIDI files (expected 6).")

        print(f"✅ Dataset initialized with {len(self.song_versions)} valid song versions.")
        # Chord encoding setup
        self.root_to_index = {
            "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4, "E#": 5,
            "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
            "B": 11, "Cb": 11, "B#": 0
        }
        self.quality_intervals = {
            # Basic triads
            "maj": ['1', '3', '5'],  # Major
            "min": ['1', 'b3', '5'],  # Minor
            "dim": ['1', 'b3', 'b5'],  # Diminished
            "aug": ['1', '3', '#5'],  # Augmented

            # Sixth chords
            "maj6": ['1', '3', '5', '6'],  # Major 6
            "min6": ['1', 'b3', '5', '6'],  # Minor 6

            # Seventh chords
            "maj7": ['1', '3', '5', '7'],  # Major 7
            "min7": ['1', 'b3', '5', 'b7'],  # Minor 7
            "minmaj7": ['1', 'b3', '5', '7'],  # Minor-Major 7
            "dim7": ['1', 'b3', 'b5', 'bb7'],  # Diminished 7 (bb7 = 9 semitones)

            # Ninth chords
            "maj9": ['1', '3', '5', '7', '9'],  # Major 9
            "min9": ['1', 'b3', '5', 'b7', '9'],  # Minor 9

            # Dominant chords
            "7": ['1', '3', '5', 'b7'],  # Dominant 7
            "9": ['1', '3', '5', 'b7', '9'],  # Dominant 9
            "11": ['1', '3', '5', 'b7', '9', '11'],  # Dominant 11

            # Suspended chords
            "sus4": ['1', '4', '5'],  # Suspended 4th (no 3rd)
            "sus2": ['1', '2', '5'],  # Suspended 2nd

            # Power chord (not technically a full chord)
            "1": ['1'],  # Root only
            "5": ['1', '5'],  # Power chord (no 3rd)
        }

        self.interval2semitones = {
            "1": 0,  # Unison
            "b2": 1,  # Minor 2nd
            "2": 2,  # Major 2nd
            "#2": 3,  # Augmented 2nd (same as minor 3rd)
            "b3": 3,  # Minor 3rd
            "3": 4,  # Major 3rd
            "4": 5,  # Perfect 4th
            "#4": 6,  # Augmented 4th (same as diminished 5th)
            "b5": 6,  # Diminished 5th
            "5": 7,  # Perfect 5th
            "#5": 8,  # Augmented 5th (same as minor 6th)
            "b6": 8,  # Minor 6th
            "6": 9,  # Major 6th
            "#6": 10,  # Augmented 6th (same as minor 7th)
            "b7": 10,  # Minor 7th
            "7": 11,  # Major 7th
            "8": 12,  # Octave
            "b9": 13,  # Minor 9th (same as b2 an octave higher)
            "9": 14,  # Major 9th (same as 2 an octave higher)
            "#9": 15,  # Augmented 9th (same as minor 3rd an octave higher)
            "11": 17,  # Perfect 11th (same as 4 an octave higher)
            "#11": 18,  # Augmented 11th (same as #4 an octave higher)
            "b13": 20,  # Minor 13th (same as b6 an octave higher)
            "13": 21,  # Major 13th (same as 6 an octave higher)
        }

    def convert_interval(self, interval):
        counter = 0
        for i in interval:
            if i == '#':
                counter += 1
            elif i == 'b':
                counter -= 1
            else:
                return (self.interval2semitones[i] + counter)

    def encode_chord(self, chord):
        """ Encodes a chord into root, quality, and inversion. """
        key, part = chord.split(':')
        parts = part.split("/")

        encoded_root = self.root_to_index[key]
        quality = parts[0]

        if len(parts) == 2:
            inversion_in_semitones = self.convert_interval(parts[1])
            encoded_bass = (encoded_root + inversion_in_semitones) % OCTAVE
        else:
            encoded_bass = encoded_root


        bass_vec = np.zeros(OCTAVE)
        bass_vec[encoded_root] = 1

        # One-hot encode root
        root_vec = np.zeros(OCTAVE)
        root_vec[encoded_root] = 1

        # Convert quality to intervals
        pitches_vec = np.zeros(OCTAVE)
        pitches_vec[encoded_root] = pitches_vec[encoded_bass] = 1

        # Extract base quality and extensions inside ()
        base_quality = re.sub(r"\(.*?\)", "", quality)  # Remove parentheses
        extensions = re.findall(r"\((.*?)\)", quality)  # Extract contents inside ()
        assert base_quality != '' or extensions != '', f"'{chord}' acts weird"

        if extensions:
            for ext in extensions[0].split(","):
                ext = ext.strip()
                semitone_value = self.convert_interval(ext)
                pitches_vec[semitone_value % OCTAVE] = 1

        if base_quality:
            if base_quality in self.quality_intervals:
                intervals = self.quality_intervals[base_quality]
                for interval in intervals:
                    interval_in_semitones = self.convert_interval(interval)
                    pitches_vec[interval_in_semitones % OCTAVE] = 1
            else:
                raise ValueError(f"quality: '{base_quality}' not supported (yet). chord: '{chord}'")

        return np.concatenate([root_vec, pitches_vec, bass_vec])

    def process_midi(self, midi_files):
        """ Combines multiple MIDI files into a single numpy array. """
        midi_to_fret_converter = MidiToFret()
        processed_midi, uncertainty_mask = midi_to_fret_converter.process_midi(midi_files)
        non_silent_midi = midi_to_fret_converter.midi_to_fret(processed_midi)

        return non_silent_midi, uncertainty_mask

    def show_sample(self, index=None):
        """Displays a sample input-output pair from the dataset."""
        if index is None:
            index = random.randint(0, len(self) - 1)

        chords, midi_prev, midi_target, song_id, version, textual_chords = self[index]

        print(f"\n🎵 **Sample from song_id: {song_id}, version: {version}** 🎵")
        print("\n🔹 **Chord Sequence (Numerical Encoding)**:")
        print(textual_chords)
        print(chords.numpy().shape)
        print("root encoding")
        print(chords[:,:12])
        print("pitches encoding:")
        print(chords[:,12:24])
        print("bass_encoding:")
        print(chords[:,24:])
        print(chords.numpy())

        print("\n🎼 **MIDI Input Representation (t-1):**")
        print(midi_prev.numpy().shape)
        print(midi_prev.numpy())

        print("\n🎯 **MIDI Target Representation (t):**")
        print(midi_target.numpy().shape)
        print(midi_target.numpy())

        print("\n✅ **Example Input-Output Pair Structure:**")
        print(f"INPUT: (Chords until t, MIDI until t-1) -> OUTPUT: MIDI at t")

    def __len__(self):
        return len(self.song_versions)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        a batch will have frets_prev shape (batch, 6 strings + uncertainty, history_len, frets)
        """
        song_id, version = self.song_versions[idx]

        # Load chord labels
        lab_path = os.path.join(self.lab_dir, f"{song_id}_full_done.lab")
        with open(lab_path, "r") as f:
            chords = [line.strip().split("\t")[-1] for line in f.readlines() if line != '']

        encoded_chords = np.array([self.encode_chord(ch) for ch in chords])

        # Load corresponding MIDI files for this song/version
        midi_files = sorted([
            os.path.join(self.midi_dir, f) for f in os.listdir(self.midi_dir)
            if f.startswith(f"{song_id}_{version}") and f.endswith(".mid")
        ])
        midi_data, uncertainty_mask = self.process_midi(midi_files)

        # Pick a random timestep `t` such that `t+1` is within range
        t = np.random.randint(self.history_length, midi_data.shape[1] - 1)

        # Extract uncertainty mask (7th row)
        uncertainty_mask = uncertainty_mask[t]  # Shape: (6,)

        # Prepare input history of `history_length` timesteps
        chords_input = encoded_chords[t - self.history_length:t+1]  # (history_length, chord_features)
        frets_prev_input = midi_data[:6, t - self.history_length:t, :]  # (6, history_length, 23)

        # Target is only for the next timestep `t+1`
        frets_target = midi_data[:6, t ]  # Shape: (6,)

        return torch.tensor(chords_input, dtype=torch.float32), \
               torch.tensor(frets_prev_input, dtype=torch.float32), \
               torch.tensor(frets_target, dtype=torch.float32), \
               torch.tensor(uncertainty_mask, dtype=torch.float32), \
               song_id,\
                version, \
                chords




import pretty_midi
import numpy as np

class MidiToFret:
    def __init__(self, tuning=[40, 45, 50, 55, 59, 64], fs=100):  # Standard tuning EADGBE
        self.tuning = tuning  # MIDI note numbers for the open strings
        self.fs = fs

    def process_midi(self, midi_files):
        """ Combines multiple MIDI files into a single numpy array. """
        midi_data = [pretty_midi.PrettyMIDI(f) for f in midi_files]

        # Find max sequence length among all MIDI files
        max_time_steps = max([m.get_piano_roll(fs=self.fs).shape[1] for m in midi_data])

        # Initialize a (6, max_time_steps) array with zeros
        midi_array = np.zeros((6, max_time_steps))

        # Fill in actual MIDI data
        for i, midi in enumerate(midi_data):
            piano_roll = midi.get_piano_roll(fs=self.fs)
            time_steps = piano_roll.shape[1]

            uncertainty_array = np.zeros(time_steps) if i == 5 else None
            # Extract the highest active MIDI note per timestep
            midi_array[i, :time_steps] = [
                np.max(np.where(piano_roll[:, t] > 0)[0]) if (np.any(piano_roll[:, t] > 0) and not np.any(piano_roll[35, t] > 0)) else 0
                for t in range(time_steps)
            ]

            if i == 5:
                uncertainty_array[:time_steps] = [
                    1 if np.any(piano_roll[35, t] > 0) else 0
                    for t in range(time_steps)
                ]

        # Compute the shortest note duration
        shortest_note_duration = self.get_shortest_note_duration(midi_data)

        # Segment the MIDI array based on the shortest note duration
        segmented_midi, uncertainty_array = self.segment_by_shortest_note(midi_array, uncertainty_array, shortest_note_duration)


        # Remove silent columns
        non_silent_midi, uncertainty_array, silent_indices = self.remove_silent_timesteps(segmented_midi, uncertainty_array)

        return non_silent_midi, uncertainty_array

    def remove_silent_timesteps(self, midi_array, uncertainty_array):
        """Removes columns where all 6 rows are zero."""
        non_silent_indices = np.any(midi_array != 0, axis=0)  # True where any note is played
        non_silent_midi = midi_array[:, non_silent_indices]  # Keep only non-silent columns
        non_silent_uncertainty = uncertainty_array[non_silent_indices]

        silent_indices = np.where(~non_silent_indices)[0]  # Indices of removed silent timesteps

        return non_silent_midi, non_silent_uncertainty, silent_indices

    def get_shortest_note_duration(self, midi_data):
        """Finds the shortest note duration in the MIDI files."""
        min_duration = float("inf")

        for midi in midi_data:
            for instrument in midi.instruments:
                for note in instrument.notes:
                    duration = note.end - note.start
                    if duration < min_duration:
                        min_duration = duration

        return min_duration

    def segment_by_shortest_note(self, midi_array, uncertainty_array, shortest_duration):
        """
        Segments the MIDI array into chunks of `shortest_duration` time intervals.
        """
        num_timesteps = midi_array.shape[1]
        segment_size = max(1, int(shortest_duration * self.fs))  # Convert duration to timesteps
        num_segments = num_timesteps // segment_size

        segmented = []
        segmented_uncertainty = []
        segment_map = []

        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, num_timesteps)

            # Average over the chunk to preserve note intensities
            segment = np.mean(midi_array[:, start_idx:end_idx], axis=1)
            segment_uncert = np.mean(uncertainty_array[start_idx:end_idx])
            segmented.append(segment)
            segmented_uncertainty.append(segment_uncert)
            segment_map.append(segment_size)

        return np.array(segmented).T, np.array(segmented_uncertainty).T


    def midi_to_fret(self, midi_array):
        """Converts MIDI note numbers to fret numbers and one-hot encodes them."""

        num_strings, num_timesteps = midi_array.shape
        max_frets = 25  # 0-21 frets, +1 for muted (-1)

        # Initialize fret array (-1 means no note)
        fret_array = np.full((num_strings, num_timesteps), -1, dtype=int)

        # One-hot encoded output: (strings, timesteps, frets+1)
        one_hot_fret = np.zeros((num_strings, num_timesteps, max_frets + 1), dtype=int)

        for string_idx, open_note in enumerate(self.tuning):
            for time_idx in range(num_timesteps):
                midi_note = int(round(midi_array[string_idx, time_idx]))  # Round MIDI to integer

                if midi_note != 0:  # A note is played
                    fret_number = midi_note - open_note + 1 # plus 1 for offset of muted being index 0

                    if 1 <= fret_number <= max_frets:  # Valid fret range
                        fret_array[string_idx, time_idx] = fret_number
                        one_hot_fret[string_idx, time_idx, fret_number] = 1
                    else:
                        print(
                            f"⚠️ Warning: MIDI {midi_note} out of fret range on string {string_idx} at time {time_idx}")
                else:
                    # No note is played, mark as muted (-1)
                    one_hot_fret[string_idx, time_idx, 0] = 1

       # return fret_array, one_hot_fret
        return one_hot_fret


class DatasetStats(FretboardFlow):
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


        self._process_dataset()

    def _process_dataset(self):
        output_records = []
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

            for t in range(encoded_chords.shape[0]):
                output_records.append({
                    "song_id": song_id,
                    "version": version,
                    "chord_index_in_song": t,
                    "chord_label": chord_labels[t],
                    "voicing": str(torch.argmax(torch.Tensor(midi_data[:,t,:]).squeeze(), dim=-1).numpy())
                })
            # 3) For each valid time t in [history_length, T-1],
            #    build a sample. This ensures we get slices for *every* t
            #    across the progression, not just a random one.
            # for t in range(self.history_length, T - 1):
            #     # Extract the portion for this sample
            #     sample = self._build_sample(
            #         encoded_chords, chord_labels,
            #         midi_data, uncertainty_mask,
            #         t, song_id, version
            #     )
            #     if sample is not None:
            #         self.fullseq_samples.append(sample)

        df = pd.DataFrame(output_records)
        df.to_csv("voiced_chord_sequence.csv", index=False)
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


dataset = DatasetStats('.', history_length=7)
