import torch

from .config import NUM_FRETS

import torch.nn.functional as F

def brier_score_per_string(pred_logits, target_one_hot):
    """
    Computes the Brier score for a chord-like prediction,
    where predictions and targets are shaped (n_strings, n_frets).

    Args:
        pred_logits (Tensor): Raw model outputs, shape (n_strings, n_frets)
        target_one_hot (Tensor): One-hot target labels, shape (n_strings, n_frets)

    Returns:
        float: Average Brier score across strings
    """
    # Step 1: Apply softmax per string (across frets)
    probs = F.softmax(pred_logits, dim=-1)  # softmax over frets per string

    # Step 2: Compute squared error between predicted probs and one-hot labels
    brier_score = torch.mean((probs - target_one_hot) ** 2)

    return brier_score


def pitch_class_loss(output_probs, frets_target, last_chord_textual):
    """
    Loss function that only punishes predictions that do not match the pitch class of the target.

    Args:
        output_probs: Softmax probabilities of shape (batch, 6, 23) (probabilities per fret per string)
        frets_target: One-hot encoded ground truth of shape (batch, 6, 23)
        padding_mask: Mask of shape (batch, 6, 23) to ignore padded timesteps

    Returns:
        Scalar loss penalizing only incorrect pitch-class predictions.
    """
    # Convert one-hot target to fret index (batch, 6)
    # frets_target_indices = torch.argmax(frets_target, dim=-1)  # Shape: (batch, 6)
    begin_pitch_vector = 12
    end_pitch_vector = 24

    target_pitch_classes = torch.tensor(last_chord_textual[:, -1 ,begin_pitch_vector:end_pitch_vector], \
                                        device=output_probs.device)
    tuning = torch.tensor([40, 45, 50, 55, 59, 64], device=output_probs.device).unsqueeze(0).unsqueeze(-1)  # Guitar tuning in MIDI

    all_frets = torch.arange(NUM_FRETS+1, device=output_probs.device).view(1, 1, -1)  # Shape: (1, 1, 23)
    pred_pitch_classes = (tuning + all_frets) % 12  # Shape: (1, 6, 23)


    pitch_class_penalty = torch.tensor([0,0,0,0,0,0]).unsqueeze(0).float()

    for i in range(last_chord_textual.shape[0]):
        cur_target = target_pitch_classes[i]
        pitch_classes = torch.nonzero(cur_target)

        pitch_class_mask = (pred_pitch_classes.unsqueeze(-1) == pitch_classes.flatten()).any(-1).float()

        pitch_class_penalty += torch.log1p(((1 - pitch_class_mask) * output_probs[i,:,:]).sum(dim=-1))  # Zero out correct pitches

    return pitch_class_penalty.sum()


def fret_boundary_loss(input_torch, target):
    """
    Computes a boundary-based loss for guitar fret prediction.
    Penalizes predictions based on their distance from the target fret.

    Args:
        input_torch (Tensor): Model output logits (batch, 6, 23).
        target (Tensor): Target fret positions (batch, 6).

    Returns:
        Tensor: Scalar boundary loss.
    """
    input_torch = torch.sigmoid(input_torch)  # Convert logits to probabilities

    # Convert target to one-hot encoding
    target_one_hot = torch.zeros_like(input_torch)  # Shape: (batch, 6, 23)
    target_one_hot.scatter_(-1, target.unsqueeze(-1), 1)  # One-hot encode the target

    # Compute absolute distance from the true fret
    loss = torch.zeros_like(input_torch[:, :, 0])  # Shape: (batch, 6)
    for index in range(input_torch.size(-1)):  # Iterate over 23 frets
        loss += input_torch[:, :, index] * torch.abs(target - index)

    return torch.mean(loss)  # Final scalar loss

