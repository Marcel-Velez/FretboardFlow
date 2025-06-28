import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import re
import torch
import os
import pickle as pkl

from collections import Counter

class ChordProgressionDatasetAugmented(Dataset):
    def __init__(self, root_dir, csv_path='./dadagp_own_chordnaming.csv', history_length=3, tuning=[40, 45, 50, 55, 59, 64],
                 cache_dir='./data'):
        """
        Args:
            csv_path (str): Path to the CSV file.
            history_length (int): Number of previous timesteps to include in input.
        """
        self.history_length = history_length
        self.data = pd.read_csv(csv_path)

        # Group rows by song ("filename" column)
        self.songs = self.data.groupby("filename")


        self.tuning = tuning  # MIDI note numbers for the open strings

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
            "Maj9": ['1', '3', '5', '7', '9'],  # Major 9
            "9#": ['1', '3', '5', '7', '#9'],  # Major 9
            "min9": ['1', 'b3', '5', 'b7', '9'],  # Minor 9

            #add dadagp chord
            "7sus2": ['1', '2', '5', 'b7'],  # Dominant 7
            "7sus4": ['1', '4', '5', 'b7'],  # Dominant 7

            # Dominant chords
            "2": ['1', '2', '3', '5'],  # Dominant 7
            "6": ['1', '3', '5', '6'],  # Dominant 7
            "7": ['1', '3', '5', 'b7'],  # Dominant 7
            "9": ['1', '3', '5', 'b7', '9'],  # Dominant 9
            "11": ['1', '3', '5', 'b7', '9', '11'],  # Dominant 11
            "13": ['1', '3', '5', 'b7', '9', '11', '13'],  # Dominant 11

            # Suspended chords
            "sus4": ['1', '4', '5'],  # Suspended 4th (no 3rd)
            "sus2": ['1', '2', '5'],  # Suspended 2nd

            # Power chord (not technically a full chord)
            "1": ['1'],  # Root only
            "5": ['1', '5'],  # Power chord (no 3rd)
            "5#": ['1', '5#'],  # Power chord (no 3rd)
            "#5": ['1', '#5'],  # Power chord (no 3rd)

            # weird cases
            "maj79": ['1', '3', '5', '7', '9'],  # Major 9
            "79": ['1', '3', '5', 'b7', '9'],  # Major 9
            "749": ['1', '3', '5', 'b7', '4', '9'],  # Major 9
            "min7M": ['1', 'b3', '5', '7'],  # Minor-Major 7
            "7#11": ['1', 'b3', '5', '7#', '9', '11'],  # Minor-Major 7
            "7dim": ['1', 'b3', 'b5', 'bb7'],  # Diminished 7 (bb7 = 9 semitones)

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

        self.processed_data_path = os.path.join(cache_dir, f"processed_dadagp_full_aug_data_hlen_{history_length}.pkl")

        # Check if processed data already exists
        if os.path.exists(self.processed_data_path):
            print("Loading preprocessed dataset...")
            with open(self.processed_data_path, "rb") as f:
                self.augmented_samples = pkl.load(f)
        else:
            print("Processing dataset from scratch...")
            self._generate_samples()
            # Save for future use
            with open(self.processed_data_path, "wb") as f:
                pkl.dump(self.augmented_samples, f)

        # Precompute all possible (input, target) pairs
        # self.samples = self._generate_samples()

        print(f"✅ Dataset initialized with {len(self.augmented_samples)} samples.")


    def convert_add_to_parentheses(self, chord):
        """Convert 'addX' to '(X)' in chord names, merging multiple occurrences.
           If the chord is just 'Root:addX', convert to 'Root:maj(X)'.
           If the chord contains a '/', ensure that addX appears before '/'.
           Preserve 'add11#' or 'add11b' with # or b in parentheses.
        """
        # Find all 'add' numbers including those with '#' or 'b' (e.g., add11#, add11b)
        matches = re.findall(r'add(\d+[#b]?)', chord)

        if matches:
            unique_nums = sorted(set(matches), key=matches.index)  # Remove duplicates, keep order
            # Format the matches, ensuring # or b is included if present
            replacement = f"({','.join(unique_nums)})"

            # Remove all 'addX' occurrences from the chord base
            chord_base = re.sub(r'add(\d+[#b]?)', '', chord).strip()

            # If chord_base ends with ":" (e.g., "G:add9"), assume it's a simple major chord
            if chord_base.endswith(":") or ":" not in chord_base:
                chord_base = chord_base.rstrip(":") + ":maj"

            # Handle case where there's a '/' (like in "D:sus4add9/G")
            if '/' in chord_base:
                chord_base, rest = chord_base.split('/', 1)
                return chord_base + replacement + '/' + rest

            return chord_base + replacement

        return chord

    def convert_interval(self, interval):
        counter = 0
        interval = interval.replace('sus4', '4').replace('sus2','2')

        for i in interval:
            if i == '#':
                counter += 1
            elif i == 'b':
                counter -= 1
            else:
                try:
                    counter += self.interval2semitones[i]
                except:
                    raise ValueError(f"cant convert: {i} from {interval}")
        return counter

    def encode_chord(self, chord):
        """ Encodes a chord into root, quality, and inversion. """
        OCTAVE = 12
        chord = chord.replace('+','#').replace('-','b').replace('\\\\','\\').replace('\\\\','\\')
        chord = chord.split(' ')[0]
        if 'add' in chord:
            chord = self.convert_add_to_parentheses(chord)

        key, part = chord.split(':')
        parts = part.split("/")

        encoded_root = self.root_to_index[key]
        quality = parts[0]

        if len(parts) == 2:
            if len(parts[1]) == 1:
                if parts[1].upper() in self.root_to_index:
                    inversion_in_semitones = self.root_to_index[parts[1].upper()]
                    encoded_bass = inversion_in_semitones #(encoded_root + inversion_in_semitones) % OCTAVE
                else:
                    inversion_in_semitones = self.convert_interval(parts[1])
                    encoded_bass = (encoded_root + inversion_in_semitones) % OCTAVE
            else:
                encoded_bass = 0
                for subpart in parts[1]:

                    if subpart.upper() in self.root_to_index:
                        encoded_bass += self.root_to_index[subpart.upper()]
                    elif subpart == '#':
                        encoded_bass += 1
                    elif subpart == 'b':
                        encoded_bass -= 1
                    else:
                        try:
                            inversion_in_semitones = self.convert_interval(subpart)
                            encoded_bass = (encoded_root + inversion_in_semitones)
                        except:
                            raise ValueError(f"difficulty with: {parts} subpart {subpart}")
            encoded_bass = encoded_bass % OCTAVE
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


    def _generate_samples(self):
        """
        Extracts all possible (history, target) pairs from each song.
        """
        samples = []
        self.augmented_samples = []
        samples = []
        solfege_chords = ["Do", "Re", "Sol", "La", "Si", "D:o"]
        dont_include = ['jeff', 'barr', '°', 'º', '\\', '7#11', '#11', 'a#m','V','I','O', '7#4', '7b9', '7+4', '7+11',
                        '+11','a+m', '713', '79-','5th','A5+','FA','F:A', 'sus4#C', 'normal', 'Non','min2','7#9','7+9','79#','79+'
                        'min75b','min75-','min79','RE'] + solfege_chords

        for filename, song_df in self.songs:
            song_df = song_df.sort_values("measure")  # Ensure rows are in order
            num_rows = len(song_df)

            # Extract history_length + target pairs
            for t in range(self.history_length, num_rows - 1):  # Stop at second-last row
                chord_history = song_df.iloc[t - self.history_length: t+1]["chordname"].tolist()
                position_history = song_df.iloc[t - self.history_length: t]["position"].tolist()
                target_position = song_df.iloc[t]["position"]  # Target at t

                # **Skip t if any element in history or target is None**
                if any(pd.isna(chord_history)) or any(pd.isna(position_history)) or pd.isna(target_position):
                    continue

                if any(any(solfege in chord for solfege in dont_include) for chord in chord_history):
                    continue  # Skip this progression
                if any(any(solfege in chord.lower() for solfege in dont_include) for chord in chord_history):
                    continue  # Skip this progression

                if any( chord.endswith('sus') for chord in chord_history):
                    continue  # Skip this progression

                if any('\\' in [i for i,j in Counter(chord).items() if j>1] for chord in chord_history):
                    continue
                if any('/' in [i for i,j in Counter(chord).items() if j>1] for chord in chord_history):
                    continue

                try:
                    for chord in chord_history:
                        self.encode_chord(chord)
                except Exception as e:
                    print(chord_history, e)
                    continue


                samples.append({
                    "chords": chord_history,  # List of previous chords
                    "positions": position_history,  # List of previous positions
                    "target": target_position,  # The next position
                    "filename": filename,  # Keep track of the song
                })
                chords_input = np.array([self.encode_chord(ch) for ch in chord_history])
                frets_prev_input = [self.encode_textual_diagram(dia) for dia in position_history]
                frets_target = self.encode_textual_diagram(target_position)

                augmentable_sample = (
                        torch.tensor(chords_input, dtype=torch.float32),
                        torch.tensor(frets_prev_input, dtype=torch.float32),
                        torch.tensor(frets_target,    dtype=torch.float32),
                    )
                if augmentable_sample is not None:
                    # Add the original sample
                    self.augmented_samples.append(augmentable_sample)

                    # Generate downward shifts
                    self._generate_downward_shifts(augmentable_sample)

                    # Generate upward shifts
                    self._generate_upward_shifts(augmentable_sample)


    def encode_textual_diagram(self, textual_diagram):
        """Converts MIDI note numbers to fret numbers and one-hot encodes them."""

        num_strings, num_timesteps = len(textual_diagram.split('.')), len(textual_diagram)
        assert num_strings == 6, f"num strings is not 6 like a guitar but: {num_strings} {textual_diagram}"
        max_frets = 25  # 0-25 frets, +1 is done on line 251+1 for muted (-1) (25 from dhooge paper, own handles 22)

        # Initialize fret array (-1 means no note)
        if isinstance(textual_diagram[0], str):
            fret_array = np.full((num_strings), -1, dtype=int)
            one_hot_fret = np.zeros((num_strings, max_frets + 1), dtype=int)
        else:
            fret_array = np.full((num_strings, num_timesteps), -1, dtype=int)
            # One-hot encoded output: (strings, timesteps, frets+1)

        for string_idx, open_note in enumerate(self.tuning):
            if num_timesteps > 1 and not isinstance(textual_diagram[0], str):
                for time_idx in range(num_timesteps):

                    midi_note = textual_diagram[time_idx].split('.')[string_idx]  # Round MIDI to integer

                    if midi_note != 'x':  # A note is played
                        fret_number = int(midi_note) + 1

                        if 1 <= fret_number <= max_frets:  # Valid fret range
                            fret_array[string_idx, time_idx] = fret_number
                            one_hot_fret[string_idx, time_idx, fret_number] = 1
                        else:
                            print(
                                f"⚠️ Warning: MIDI {midi_note} out of fret range on string {string_idx} at time {time_idx}")
                    else:
                        # No note is played, mark as muted (-1)
                        one_hot_fret[string_idx, time_idx, 0] = 1

            else:
                midi_note = textual_diagram.split('.')[string_idx]  # Round MIDI to integer

                if midi_note != 'x':  # A note is played
                    fret_number = int(midi_note) + 1

                    if 1 <= fret_number <= max_frets:  # Valid fret range
                        fret_array[string_idx] = fret_number
                        one_hot_fret[string_idx, fret_number] = 1
                    else:
                        print(
                            f"⚠️ Warning: MIDI {midi_note} out of fret range on string {string_idx} ")
                else:
                    # No note is played, mark as muted (-1)
                    one_hot_fret[string_idx, 0] = 1

        # return fret_array, one_hot_fret
        return one_hot_fret


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
        (chords_input, frets_prev_input, frets_target) = sample

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
        x, y, z = self.augmented_samples[idx]
        return x, y, z, 0
        #
        # try:
        #     chords_input = np.array([self.encode_chord(ch) for ch in sample["chords"]])
        #     frets_target = self.encode_textual_diagram(sample["target"])
        #     frets_prev_input = [self.encode_textual_diagram(dia) for dia in sample["positions"]]
        # except:
        #     raise ValueError(f"could not process item: {idx} chords: {sample['chords']} target: {sample['target']} position: {sample['positions']}")
        #
        # return torch.tensor(chords_input, dtype=torch.float32), \
        #        torch.tensor(frets_prev_input, dtype=torch.float32), \
        #        torch.tensor(frets_target, dtype=torch.float32), \
        #        torch.tensor([0])


