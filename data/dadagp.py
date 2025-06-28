import csv
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
import re

class ChordCSVHistoryDataset(Dataset):
    def __init__(self, root_dir, csv_path='./dadagp_own_chordnaming.csv', history_length=3, tuning=[40, 45, 50, 55, 59, 64]):
        """
        Args:
            csv_path (str): path to your CSV (dagadp_own_chordnaming.csv)
            history_len (int): how many previous rows to include as 'history'
        """
        super().__init__()
        self.csv_path = csv_path
        self.history_len = history_length

        # 1) Read CSV rows => group by filename in the exact order
        #    We'll store each file as a list of row dicts in the order they appear.
        self.song_rows = self._load_and_group_csv(self.csv_path)

        # 2) Build a flat list of (filename, t) samples
        #    where t is the row index in that song
        # self.samples = []
        # for filename, rows in self.song_rows.items():
        #     # For each valid t in [history_len .. len(rows)-1]
        #     # we produce a sample
        #     num_rows = len(rows)
        #     for t in range(self.history_len, num_rows):
        #         self.samples.append((filename, t))

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

    def convert_add_to_parentheses(self, chord):
        """Convert 'addX' to '(X)' in chord names, merging multiple occurrences.
           If the chord is just 'Root:addX', convert to 'Root:maj(X)'.
        """
        matches = re.findall(r'add(\d+)', chord)  # Find all 'addX' numbers
        if matches:
            unique_nums = sorted(set(matches), key=matches.index)  # Remove duplicates, keep order
            replacement = f"({','.join(unique_nums)})"
            chord_base = re.sub(r'add(\d+)', '', chord).strip()  # Remove all 'addX' occurrences

            # If chord_base ends with ":" (e.g., "G:add9"), assume it's a simple major chord
            if chord_base.endswith(":") or ":" not in chord_base:
                chord_base = chord_base.rstrip(":") + ":maj"

            return chord_base + replacement
        return chord


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
        OCTAVE = 12

        if 'add' in chord:
            chord = self.convert_add_to_parentheses(chord)

        print(chord)
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


    def _load_and_group_csv(self, csv_path):
        """
        Reads the CSV in row order.
        For each row, we store a dict of columns,
        grouped by 'filename'.

        We'll assume there's a 'filename' column,
        'chord_name', 'position', etc.
        The row's index in the CSV is effectively the time step.
        """
        song_rows = defaultdict(list)

        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)  # If your CSV has headers
            # If it doesn't have headers, use csv.reader and parse columns by index

            for row in reader:
                fn = row["filename"]
                # store the entire row in the list for that file
                song_rows[fn].append(row)

        return song_rows

    def __len__(self):
        return len(self.song_rows)

    def encode_textual_diagram(self, textual_diagram):
        """Converts MIDI note numbers to fret numbers and one-hot encodes them."""

        num_strings, num_timesteps = len(textual_diagram.split('.')), len(textual_diagram)
        assert num_strings == 6, f"num strings is not 6 like a guitar but: {num_strings} {textual_diagram}"
        max_frets = 22  # 0-21 frets, +1 for muted (-1)

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

                        if 1 <= fret_number <= 21:  # Valid fret range
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

                    if 1 <= fret_number <= 21:  # Valid fret range
                        fret_array[string_idx] = fret_number
                        one_hot_fret[string_idx, fret_number] = 1
                    else:
                        print(
                            f"⚠️ Warning: MIDI {midi_note} out of fret range on string {string_idx} at time {time_idx}")
                else:
                    # No note is played, mark as muted (-1)
                    one_hot_fret[string_idx, 0] = 1

        # return fret_array, one_hot_fret
        return one_hot_fret

    def __getitem__(self, idx):
        """
        Returns a sample for (filename, t).
        We'll gather chords from rows [t-history_len.. t],
        positions from [t-history_len.. t-1], etc.
        """
        rows = self.song_rows[list(self.song_rows.keys())[idx]]
        # rows = self.song_rows[filename]  # This is a list of row dicts in order

        # chord_context = chord_name for rows in [t-history_len.. t]
        chords = []
        for i in range(self.history_len+1):
            chords.append(rows[i]["chordname"])

        # encode chords
        print(f"{chords=}")
        chords_input = np.array([self.encode_chord(ch) for ch in chords])

        # position_context = position for rows in [t-history_len.. t-1]
        frets_input = []
        for i in range( self.history_len+1):
            frets_input.append(rows[i].get("position"))

        frets_input = [self.encode_textual_diagram(dia) for dia in frets_input]

        frets_input = torch.tensor(frets_input)
        #some translation from textual to our chords

        frets_target = frets_input[-1]
        frets_prev_input = frets_input[:-1]


        # If you want to transform chord strings to embeddings or one-hot,
        # you can do it here. For example:
        # chord_context_enc = [self.encode_chord(ch) for ch in chord_context]
        # chord_context_enc = torch.tensor(chord_context_enc, dtype=torch.float32)

        return torch.tensor(chords_input, dtype=torch.float32), \
               torch.tensor(frets_prev_input, dtype=torch.float32), \
               torch.tensor(frets_target, dtype=torch.float32), \
               None, \
               None,\
                None, \
                chords
