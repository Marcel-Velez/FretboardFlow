import unittest
import torch

from torch.utils.data import DataLoader
from data.dataset import FretboardFlow
from models.discontinued.lstm import GuitarFretPredictor


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    batch = [b for b in batch if b is not None]  # Remove None samples

    if len(batch) == 0:
        return None  # If no valid samples remain, return None

    chords_batch, frets_prev_batch, frets_target_batch, uncertainty_mask,  _, _, _ = zip(*batch)

    # Pad sequences to the longest in batch (if needed)
    chords_batch = torch.nn.utils.rnn.pad_sequence(chords_batch, batch_first=True, padding_value=0)
    frets_prev_batch = torch.nn.utils.rnn.pad_sequence(frets_prev_batch, batch_first=True, padding_value=0)
    frets_target_batch = torch.stack(frets_target_batch)
    frets_target_mask = torch.stack(uncertainty_mask)

    return chords_batch, frets_prev_batch, frets_target_batch, frets_target_mask



class TestModelDataLoaderCompatibility(unittest.TestCase):
    def setUp(self):
        """Set up a small dataset and model for compatibility testing."""
        self.dataset = FretboardFlow(root_dir="data/")  # Replace with actual dataset path
        self.dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

        # Initialize model
        self.model = GuitarFretPredictor(
            hidden_size=128,
            num_layers=2
        )

    def test_dataloader_shapes(self):
        """Ensure dataset provides correctly shaped batches."""
        batch = next(iter(self.dataloader))  # Fetch a batch
        print(f"{len(batch)=}")

        chords, frets_prev, frets_target, uncertainty_mask = batch

        print(f"{len(uncertainty_mask)}")
        # Check batch dimensions
        batch_size, history, chord_features = chords.shape
        print(f"{frets_prev.shape=}")
        batch_size_frets, n_strings_and_uncertainty, history_frets, position_features = frets_prev.shape

        self.assertEqual(chord_features, self.model.chord_lstm.input_size,
                         f"Chord feature size mismatch: expected {self.model.chord_lstm.input_size}, got {chord_features}")

        self.assertEqual(position_features * (n_strings_and_uncertainty), self.model.position_lstm.input_size,
                         f"Position feature size mismatch: expected {self.model.position_lstm.input_size}, got {position_features}")

        # Check uncertainty mask shape
        self.assertEqual(uncertainty_mask.shape, (batch_size,),
                         f"Uncertainty mask shape mismatch: expected {(batch_size, )}, got {uncertainty_mask.shape}")

    def test_model_forward_pass(self):
        """Ensure model runs without shape mismatches."""
        batch = next(iter(self.dataloader))
        chords, frets_prev, frets_target, uncertainty_mask = batch

        print(f"{chords.shape=} {frets_prev.shape=} {frets_target.shape=}")

        output = self.model(chords, frets_prev)

        expected_shape = (chords.shape[0], 6, 23)
        self.assertEqual(output.shape, expected_shape,
                         f"Expected output shape {expected_shape}, but got {output.shape}")

    def test_uncertainty_mask_values(self):
        """Ensure uncertainty mask contains only 0s and 1s."""
        batch = next(iter(self.dataloader))
        _, _, _, uncertainty_mask = batch

        unique_values = torch.unique(uncertainty_mask).tolist()
        invalid_values = [val for val in unique_values if val not in [0, 1]]

        self.assertTrue(len(invalid_values) == 0,
                        f"Uncertainty mask contains invalid values {invalid_values}")

if __name__ == "__main__":
    unittest.main()