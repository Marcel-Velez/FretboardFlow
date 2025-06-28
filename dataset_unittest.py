import unittest
import os
import numpy as np
import torch
from data.dataset import FretboardFlow
import random

class TestChordMIDI_Dataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Runs once before all tests. Initializes dataset."""
        cls.dataset = FretboardFlow(root_dir="data/")

        cls.dataset.show_sample()

    def test_chord_midi_length_match(self):
        """Checks that the number of chords equals the number of MIDI timesteps for all dataset entries."""
        mismatches = []  # Store mismatches for later reporting

        for i in range(len(self.dataset)):
            chords, midi_prev, midi_target, song_id, version, _ = self.dataset[i]

            # Chords length (timesteps from label file)
            chord_len = chords.shape[0]  # Number of chord timesteps

            # MIDI length (timesteps from MIDI files)
            midi_len = midi_prev.shape[1]  # Time dimension of MIDI array

            # Check if they match
            if chord_len != midi_len:
                mismatches.append(
                    f"Mismatch at index {i} | song_id: {song_id} | version: {version} | "
                    f"chords: {chord_len} vs MIDI: {midi_len}"
                )

        # If mismatches were found, report them all at once
        if mismatches:
            for error in mismatches:
                print(error)
            self.fail(f"{len(mismatches)} mismatches found in dataset.")


    def test_midi_file_count(self):
        """Ensures that each dataset entry loads exactly 6 MIDI files."""
        for i in range(len(self.dataset)):
            _, midi_prev, _, _, _, _ = self.dataset[i]

            # Check shape: should have 7 rows (6 strings + 1 uncertainty row)
            self.assertEqual(midi_prev.shape[0], 7,
                             f"Expected 7 rows (6 MIDI + 1 uncertainty), got {midi_prev.shape[0]} at index {i}")

            # Extract the uncertainty row (last row: index 6)
            uncertainty_row = midi_prev[6, :]

            # Ensure all values in the uncertainty row are 0 or 1 (binary flag)
            unique_values = np.unique(uncertainty_row)
            invalid_values = [val for val in unique_values if val not in [0, 1]]

            self.assertTrue(len(invalid_values) == 0,
                            f"Uncertainty row contains non-binary values {invalid_values} at index {i}")

    def test_chord_encoding_coverage(self):
        """Ensures that every chord in the dataset has a valid encoding mapping."""
        missing_chords = set()
        all_chords = set()

        for i in range(len(self.dataset)):
            _, _, _, song_id, version, _ = self.dataset[i]

            # Load chords from the corresponding lab file
            lab_path = os.path.join(self.dataset.lab_dir, f"{song_id}_full_done.lab")
            with open(lab_path, "r") as f:
                chords = [line.strip().split("\t")[-1] for line in f.readlines()]

            for chord in chords:
                all_chords.add(chord)  # Keep track of all unique chords

                try:
                    self.dataset.encode_chord(chord)  # Attempt encoding
                except ValueError:
                    missing_chords.add(chord)  # If encoding fails, add to missing list

        # If there are missing chords, report them
        if missing_chords:
            print(f"⚠️ {len(missing_chords)} chords are missing from encoding!")
            for chord in sorted(missing_chords):
                print(f"  - {chord}")
            self.fail(f"Missing chord mappings for {len(missing_chords)} chords in the dataset.")

        else:
            print(f"✅ All {len(all_chords)} unique chords have valid encodings!")

if __name__ == "__main__":
    unittest.main()
