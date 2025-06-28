import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import re
import torch
import os
import pickle as pkl

from collections import Counter

class ChordProgressionDataset(Dataset):
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

        self.processed_data_path = os.path.join(cache_dir, f"processed_dadagp_full_data_hlen_{history_length}.pkl")

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

        # Check if processed data already exists
        if os.path.exists(self.processed_data_path):
            print("Loading preprocessed dataset...")
            with open(self.processed_data_path, "rb") as f:
                self.samples = pkl.load(f)
        else:
            print("Processing dataset from scratch...")
            self.samples = self._generate_samples()
            # Save for future use
            with open(self.processed_data_path, "wb") as f:
                pkl.dump(self.samples, f)

        # Precompute all possible (input, target) pairs
        # self.samples = self._generate_samples()

        print(f"✅ Dataset initialized with {len(self.samples)} samples.")


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

        return samples

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


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a single sample:
        - chord history of length `history_length`
        - position history of length `history_length`
        - target position at `t`
        """
        sample = self.samples[idx]

        try:
            chords_input = np.array([self.encode_chord(ch) for ch in sample["chords"]])
            frets_target = self.encode_textual_diagram(sample["target"])
            frets_prev_input = [self.encode_textual_diagram(dia) for dia in sample["positions"]]
        except:
            raise ValueError(f"could not process item: {idx} chords: {sample['chords']} target: {sample['target']} position: {sample['positions']}")

        return torch.tensor(chords_input, dtype=torch.float32), \
               torch.tensor(frets_prev_input, dtype=torch.float32), \
               torch.tensor(frets_target, dtype=torch.float32), \
               torch.tensor([0])


