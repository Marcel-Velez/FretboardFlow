import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch
from typing import List, Tuple, Dict
import torch.nn.functional as F
import numpy as np

# Suppose we have all these from your snippet:
from .metrics import GuitarMetrics
from .anatomical import anatomical_score, string_centroid, ratio_muted_strings, ratio_open_strings
from .losses import pitch_class_loss, fret_boundary_loss, brier_score_per_string  # Example placeholders
# ... etc.

from .config import NUM_FRETS, NUM_STRINGS


class BaseGuitarTrainer(pl.LightningModule):
    """
    Generic PyTorch Lightning Module for guitar fret prediction tasks.
    Model-specific classes (LSTM, GRU, Transformer, etc.)
    should inherit from this class and implement `forward(...)`.
    """

    def __init__(self, hidden_size=128, num_layers=2, uncertainty_weight=0.5,
                 learning_rate=1e-3, fretboardflow=True, multi_fret_loss=0.0, pitch_class_loss=0.0,
                 distance_loss=0.0, brier_score=False, nllloss=False, mseloss=False, args=None):
        super().__init__()
        self.save_hyperparameters(args)
        self.uncertainty_weight = uncertainty_weight
        self.fretboardflow = fretboardflow
        self.metrics = GuitarMetrics()
        self.learning_rate = learning_rate

        self.multi_fret_loss= multi_fret_loss
        self.pitch_class_loss = pitch_class_loss
        self.distance_loss = distance_loss

        self.brier_score = brier_score
        self.nllloss = nllloss
        self.mseloss = mseloss

        self.dhooge = False
        # Example final FC layer: your derived classes might create their own or override
        # but let's define a standard one for output shape (batch, 6, NUM_FRETS)

        # For quick metric computations
        from torchmetrics import Accuracy, Precision, Recall, F1Score
        self.acc = Accuracy(task="binary")
        self.prec = Precision(task="binary")
        self.rec = Recall(task="binary")
        self.f1 = F1Score(task="binary")

    def forward(self, chords_history, frets_prev_one_hot):
        """
        This is the model forward pass.
        MUST be overridden by child classes (LSTM, GRU, etc.).
        Should return a Tensor of shape: (batch, 6, NUM_FRETS).
        """
        raise NotImplementedError("Please implement forward(...) in the child class.")

    def training_step(self, batch, batch_idx):
        """
        Common training step across models:
        1) forward pass
        2) compute loss
        3) compute & log metrics
        """
        mhe_chord_symbol_history, ohe_frets_history, ohe_frets_target, uncertainty_mask = batch

        if not self.fretboardflow:
            ohe_frets_history = ohe_frets_history.permute(0,2,1,3)


        output = self(mhe_chord_symbol_history, ohe_frets_history)

        # Compute total loss
        loss = self.compute_loss(output, ohe_frets_target, uncertainty_mask, mhe_chord_symbol_history)

        # Compute metrics
        tab_metrics = self.compute_batch_tablature_metrics(output, ohe_frets_target)


        predicted_frets = torch.argmax(output, dim=-1)
        integer_frets_target = torch.argmax(ohe_frets_target, dim=-1)

        pitch_metrics = self.compute_batch_pitch_metrics(predicted_frets, integer_frets_target)
        playability_score = self.compute_batch_playability(predicted_frets)

        # Log training metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(pitch_metrics, prog_bar=True, on_epoch=True)
        self.log_dict(tab_metrics, prog_bar=True, on_epoch=True)
        self.log("train_unplayability", playability_score, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Common validation step across models.
        """
        mhe_chord_symbol_history, ohe_frets_history, ohe_frets_target, uncertainty_mask = batch

        if not self.fretboardflow:
            ohe_frets_history = ohe_frets_history.permute(0,2,1,3)


        output = self(mhe_chord_symbol_history, ohe_frets_history)

        # Compute loss
        loss = self.compute_loss(output, ohe_frets_target, uncertainty_mask, mhe_chord_symbol_history)

        # Evaluate metrics
        predicted_frets = torch.argmax(output, dim=-1)
        integer_frets_target = torch.argmax(ohe_frets_target, dim=-1)

        pitch_metrics = self.compute_batch_pitch_metrics(predicted_frets, integer_frets_target)
        tab_metrics = self.compute_batch_tablature_metrics(output, ohe_frets_target)
        playability_score = self.compute_batch_playability(predicted_frets)

        # Possibly do transition cost & other advanced metrics
        # (some are omitted for brevity)

        probs = F.softmax(output, dim=-1)  # softmax over frets per string

        log_softmax = nn.LogSoftmax(dim=-1)
        loss_fn = nn.NLLLoss(reduction='mean')
        log_probs_flat = log_softmax(output.requires_grad_(requires_grad=False).view(-1, 26)) # [batch_size * n_strings, n_frets] → [96, 26]
        targets_flat = integer_frets_target.view(-1)  # [batch_size * n_strings] → [96]
        nll_loss = loss_fn(log_probs_flat, targets_flat)
        loss_fn = nn.MSELoss(reduction='mean')
        mse_loss = loss_fn(probs.requires_grad_(requires_grad=False), ohe_frets_target)

        self.log("validation_nll_loss", nll_loss, prog_bar=True, on_epoch=True)
        self.log("validation_mse_loss", mse_loss, prog_bar=True, on_epoch=True)


        # Additional naive metrics
        acc  = self.acc(output, ohe_frets_target)
        rec  = self.rec(output, ohe_frets_target)
        prec = self.prec(output, ohe_frets_target)
        f1   = self.f1(output, ohe_frets_target)

        self.log("val_naive_acc", acc, prog_bar=True, on_epoch=True)
        self.log("val_naive_rec", rec, prog_bar=True, on_epoch=True)
        self.log("val_naive_prec", prec, prog_bar=True, on_epoch=True)
        self.log("val_naive_f1", f1, prog_bar=True, on_epoch=True)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log_dict(pitch_metrics, prog_bar=True, on_epoch=True)
        self.log_dict(tab_metrics, prog_bar=True, on_epoch=True)
        self.log("val_unplayability", playability_score, prog_bar=True, on_epoch=True)
        prev_frets = torch.argmax(ohe_frets_history, dim=-1)  # e.g. shape (batch_size, n_strings)

        return loss

    def test_step(self, batch, batch_idx):
        mhe_chord_symbol_history, ohe_frets_history, ohe_frets_target, uncertainty_mask = batch

        if not self.fretboardflow:
            ohe_frets_history = ohe_frets_history.permute(0,2,1,3)


        output = self(mhe_chord_symbol_history, ohe_frets_history)

        # Compute loss
        loss = self.compute_loss(output, ohe_frets_target, uncertainty_mask, mhe_chord_symbol_history)


        # Evaluate metrics
        predicted_frets = torch.argmax(output, dim=-1)
        integer_frets_target = torch.argmax(ohe_frets_target, dim=-1)


        probs = F.softmax(output, dim=-1)  # softmax over frets per string

        log_softmax = nn.LogSoftmax(dim=-1)
        loss_fn = nn.NLLLoss(reduction='mean')
        log_probs_flat = log_softmax(output.requires_grad_(requires_grad=False).view(-1, 26)) # [batch_size * n_strings, n_frets] → [96, 26]
        targets_flat = integer_frets_target.view(-1)  # [batch_size * n_strings] → [96]
        nll_loss = loss_fn(log_probs_flat, targets_flat)
        loss_fn = nn.MSELoss(reduction='mean')
        mse_loss = loss_fn(probs.requires_grad_(requires_grad=False), ohe_frets_target)

        self.log("test_nll_loss", nll_loss, prog_bar=True, on_epoch=True)
        self.log("test_mse_loss", mse_loss, prog_bar=True, on_epoch=True)



        pitch_metrics = self.compute_batch_pitch_metrics(predicted_frets, integer_frets_target, addition='test_')
        tab_metrics = self.compute_batch_tablature_metrics(output, ohe_frets_target, addition='test_')
        playability_score = self.compute_batch_playability(predicted_frets)

        prev_frets = torch.argmax(ohe_frets_history, dim=-1)  # e.g. shape (batch_size, n_strings)
        transition_cost = self.compute_batch_transition_ease_real(predicted_frets, prev_frets)

        ratio_mute = self.compute_batch_ratio_muted(predicted_frets)
        ratio_open = self.compute_batch_ratio_open_strings(predicted_frets)
        unique_pc = self.compute_batch_ratio_unique_notes(predicted_frets)
        # print(f"{predicted_frets.shape=}{prev_frets.shape=} {unique_pc.shape=}")
        # print(f"{prev_frets[:, :, -1].squeeze().shape=} \n\n\n")

        fret_centroid_new = self.compute_batch_fret_centroid(predicted_frets)
        centroid_new = self.compute_batch_string_centroid(predicted_frets)

        if len(prev_frets[:,:,-1].squeeze().shape) == 1:
            # print('asdfkjalsd;jfk')
            fret_centroid_prev = self.compute_batch_fret_centroid(prev_frets.squeeze())
            centroid_prev = self.compute_batch_string_centroid(prev_frets.squeeze())  # update not to take
            unique_prev_pc = self.compute_batch_ratio_unique_notes(prev_frets.squeeze())
        else:
            # print('helllooo')
            fret_centroid_prev = self.compute_batch_fret_centroid(prev_frets[:,:,-1].squeeze())
            centroid_prev = self.compute_batch_string_centroid(prev_frets[:, :, -1].squeeze())  # update not to take
            unique_prev_pc = self.compute_batch_ratio_unique_notes(prev_frets[:,:,-1].squeeze())

        # print(unique_prev_pc)
        #
        # exit()
        centroid =  [abs(centroid_new[i] - centroid_prev[i]) for i in range(output.shape[0])]
        unique_pc =[abs(unique_pc[i] - unique_prev_pc[i]) for i in range(output.shape[0])]
        unique_pc = torch.Tensor(unique_pc).mean()

        centroid = torch.Tensor(centroid).mean()
        fret_centroid = torch.Tensor([abs(fret_centroid_new[i] - fret_centroid_prev[i]) for i in range(output.shape[0]) if fret_centroid_new[i] != -1]).mean()



        acc = self.acc(output, ohe_frets_target)
        rec = self.rec(output, ohe_frets_target)
        prec = self.prec(output, ohe_frets_target)
        f1 = self.f1(output, ohe_frets_target)

        self.log("test_naive_acc", acc, prog_bar=True, on_epoch=True)
        self.log("test_naive_rec", rec, prog_bar=True, on_epoch=True)
        self.log("test_naive_prec", prec, prog_bar=True, on_epoch=True)
        self.log("test_naive_f1", f1, prog_bar=True, on_epoch=True)

        # Log metrics
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log_dict(pitch_metrics, prog_bar=True, on_epoch=True)
        self.log_dict(tab_metrics, prog_bar=True, on_epoch=True)
        self.log("test_unplayability", playability_score, prog_bar=True, on_epoch=True)

        # Now the new metrics:
        self.log("test_transition_cost", transition_cost, prog_bar=True, on_epoch=True)
        self.log("test_ratio_mute", ratio_mute, prog_bar=True, on_epoch=True)
        self.log("test_ratio_open", ratio_open, prog_bar=True, on_epoch=True)
        self.log("test_centroid", centroid, prog_bar=True, on_epoch=True)
        self.log("test_ratio_unique", unique_pc, prog_bar=True, on_epoch=True)

        # own metric
        self.log("test_fret_centroid", fret_centroid, prog_bar=True, on_epoch=True)



    def configure_optimizers(self):
        """
        Common optimizer. Child classes can override if needed.
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    # -----------------------------
    # Example: your loss function
    # -----------------------------
    def compute_loss(self, output, fret_targets, uncertainty_mask, last_chord_textual):
        """
        Example combined cross-entropy + distance loss.
        """
        batch_size = output.shape[0]
        output = output.reshape(batch_size,NUM_STRINGS, -1)

        output_probs = torch.softmax(output, dim=-1)
        frets_target_indices = torch.argmax(fret_targets, dim=-1)

        # Cross-entropy over each string
        loss = 0.0
        if self.brier_score:
            loss = brier_score_per_string(output_probs, fret_targets)
        elif self.nllloss:
            log_softmax = nn.LogSoftmax(dim=-1)
            loss_fn = nn.NLLLoss(reduction='mean')
            log_probs_flat = log_softmax(output.view(-1, 26))
            targets_flat = frets_target_indices.view(-1)  # [batch_size * n_strings] → [96]
            loss = loss_fn(log_probs_flat, targets_flat)

        elif self.mseloss:
            loss_fn = nn.MSELoss(reduction='mean')
            loss = loss_fn(output_probs, fret_targets)
        else:
            # loss += F.binary_cross_entropy(torch.sigmoid(output), fret_targets)

            loss = 0
            for s in range(6):
                    loss += F.cross_entropy(output[:, s, :], frets_target_indices[:, s])
            loss = loss / 6.0
            # print(loss, dummy_loss)

        if self.distance_loss != 0.0:
            # Add distance penalty
            pred_fret_indices = torch.argmax(output, dim=-1)
            distances = torch.abs(pred_fret_indices - frets_target_indices)
            distance_loss = distances.float().mean()
            loss += (self.distance_loss * distance_loss)

        if self.multi_fret_loss != 0.0:
            # Suppose output_probs is shape (batch, 6, 25) after sigmoid
            multi_fret_penalty = 0
            for s in range(6):
                sum_for_string = output_probs[:, s, :].sum(dim=-1)  # shape (batch,)
                # For perfect single-fret usage, sum_for_string ≈ 1.0
                multi_fret_penalty += F.relu(sum_for_string - 1.0)  # penalize anything > 1
            multi_fret_penalty = multi_fret_penalty.mean()
            loss += (self.multi_fret_loss * multi_fret_penalty)

        if self.pitch_class_loss != 0.0:
            loss += (self.pitch_class_loss * pitch_class_loss(output_probs, fret_targets, last_chord_textual))

        return loss

    # -----------------------------
    # Example Batching Helpers
    # -----------------------------
    def compute_batch_pitch_metrics(self, batch_preds, batch_targets, addition=''):
        """Batch processing for pitch metrics (PP, RP, F1P)."""
        PP, RP, F1P = [], [], []
        for pred, gt in zip(batch_preds, batch_targets):
            p, r, f1 = self.metrics.compute_pitch_metrics(pred.cpu().numpy(), gt.cpu().numpy())
            PP.append(p)
            RP.append(r)
            F1P.append(f1)
        return {addition+"PP": np.mean(PP), addition+"RP": np.mean(RP), addition+"F1P": np.mean(F1P)}

    def compute_batch_tablature_metrics(self, batch_preds, batch_targets, addition=''):
        """Batch processing for tablature metrics (PSF, RSF, F1SF)."""
        PSF, RSF, F1SF = [], [], []
        for pred, gt in zip(batch_preds, batch_targets):
            psf, rsf, f1sf = self.metrics.compute_tablature_metrics(pred.cpu(), gt.cpu())
            PSF.append(psf)
            RSF.append(rsf)
            F1SF.append(f1sf)
        return {addition+"PSF": np.mean(PSF), addition+"RSF": np.mean(RSF), addition+"F1SF": np.mean(F1SF)}

    def compute_batch_playability(self, batch_preds):
        """
        Example for computing a playability metric in batch.
        """
        scores = []
        for chord in batch_preds:
            # Convert to your textual diagram, then compute anatomical_score or similar
            newchord = self.convert_to_dhooge_diagram(chord)
            sc = anatomical_score(newchord)[0]
            scores.append(sc)

        playability_scores = torch.tensor(scores, device=batch_preds.device)
        return (playability_scores < 0.2).float().mean()

    def convert_to_dhooge_diagram(self, chord):
        return '.'.join(str(x.item() - 1) if x != 0 else 'x' for x in chord)


    def compute_batch_ratio_muted(self, predicted_frets):
        """
        predicted_frets: (batch_size, n_strings)
        Returns average ratio of muted strings across the batch.
        """
        scores = []
        for chord in predicted_frets:
            newchord = self.convert_to_dhooge_diagram(chord)
            scores.append(ratio_muted_strings(newchord))

        return torch.tensor(scores, device=predicted_frets.device).mean()

    def compute_batch_playability(self, batch_preds):
        scores = []
        for chord in batch_preds:
            newchord = self.convert_to_dhooge_diagram(chord)
            scores.append(anatomical_score(newchord)[0])

        """Batch processing for playability metric."""
        playability_scores = torch.tensor(scores)

        return (playability_scores < 0.2).mean(dtype=torch.float)  # % of playable chords

    def compute_batch_transition_ease_real(self, predicted_frets, previous_frets):
        """
        predicted_frets: (batch_size, n_strings)
        Return: average of the chord-level anatomical_score across the batch.
        """
        scores = []
        bs = predicted_frets.shape[0]
        for i in range(bs):
            single_output = predicted_frets[i]
            single_previous = previous_frets[i,-1]
            score = self.compute_transition_ease_real(single_output, single_previous )
            scores.append(score)

        return torch.tensor(scores, device=predicted_frets.device).mean()


    def compute_transition_ease_real(self, predicted_frets, previous_frets):
        from .anatomical import anatomical_score

        """
        predicted_frets: (batch_size, n_strings)
        Return: average of the chord-level anatomical_score across the batch.
        """
        predicted_dhooge_chord = self.convert_to_dhooge_diagram(predicted_frets)
        previous_dhooge_chord = self.convert_to_dhooge_diagram(previous_frets)

        best_finger_placement = anatomical_score(predicted_dhooge_chord)[1]
        best_finger_placement_previous = anatomical_score(previous_dhooge_chord)[1]
        transition_cost = self.transition_cost(best_finger_placement,best_finger_placement_previous )
        return transition_cost

    def wrist_movement(self, fingering1: Dict[int, List[Tuple[int, int]]],
                       fingering2: Dict[int, List[Tuple[int, int]]]) -> int:
        index1 = fingering1[1][0][1]
        index2 = fingering2[1][0][1]
        return abs(index1 - index2)

    def manhattan_distance(self, v1: Tuple[int, int], v2: Tuple[int, int]) -> int:
        string_dist = abs(v1[0] - v2[0])
        fret_dist = abs(v1[1] - v2[1])
        return string_dist + fret_dist

    def finger_movement(self, fingering1: Dict[int, List[Tuple[int, int]]],
                        fingering2: Dict[int, List[Tuple[int, int]]]) -> int:
        mvt = 0
        for finger in fingering1.keys():
            if finger in fingering2.keys():
                for sf_pair1 in fingering1[finger]:
                    for sf_pair2 in fingering2[finger]:
                        dist = self.manhattan_distance(sf_pair1, sf_pair2)
                        mvt += dist
        return mvt

    def transition_cost(self, fingering1: Dict[int, List[Tuple[int, int]]],
                        fingering2: Dict[int, List[Tuple[int, int]]],
                        theta1: float = 0.25, theta2: float = 0.5) -> float:
        if fingering1 is None or fingering2 is None:
            return 0.
        a1 = self.wrist_movement(fingering1, fingering2)
        a2 = self.finger_movement(fingering1, fingering2)
        return 1 / (1 + (theta1 * a1) + (theta2 * a2))




    def compute_batch_ratio_open_strings(self, predicted_frets):
        """
        predicted_frets: (batch_size, n_strings)
        Return: average of the chord-level anatomical_score across the batch.
        """

        scores = []
        bs = predicted_frets.shape[0]
        for i in range(bs):
            chord_list = self.convert_to_dhooge_diagram(predicted_frets[i])  # e.g. [0, 2, -1, 3, 0, -1]
            score = ratio_open_strings(chord_list)
            scores.append(score)
        return torch.tensor(scores, device=predicted_frets.device).float().mean()


    def compute_batch_string_centroid(self, predicted_frets):
        """
        predicted_frets: (batch_size, n_strings)
        Return: average of the chord-level anatomical_score across the batch.
        """
        scores = []
        # print(f"{predicted_frets.shape=}")
        bs = predicted_frets.shape[0]
        for i in range(bs):
            try:
                chord_list = self.convert_to_dhooge_diagram(predicted_frets[i])  # e.g. [0, 2, -1, 3, 0, -1]
            except:
                print(predicted_frets[i,:])
                print(i)
                print(predicted_frets)
                exit()
            # print(chord_list)
            score = string_centroid(chord_list)
            scores.append(score)
        return torch.tensor(scores, device=predicted_frets.device)

    def compute_batch_ratio_unique_notes(self, predicted_frets):
        """
        predicted_frets: (batch_size, n_strings)
        Return: average of the chord-level anatomical_score across the batch.
        """
        scores = []
        bs = predicted_frets.shape[0]
        for i in range(bs):
            chord_list = predicted_frets[i].tolist()  # e.g. [0, 2, -1, 3, 0, -1]
            score = self.metrics.ratio_unique_notes_pc(chord_list)
            scores.append(score)
        return torch.tensor(scores, device=predicted_frets.device)

    def compute_batch_fret_centroid(self, predicted_frets):
        """
        predicted_frets: (batch_size, n_strings)
        Return: average of the chord-level anatomical_score across the batch.
        still have to divide by num played strings
        """
        scores = []
        bs = predicted_frets.shape[0]
        for i in range(bs):
            chord_list = self.convert_to_dhooge_diagram(predicted_frets[i])  # e.g. [0, 2, -1, 3, 0, -1]
            played_strings, fret_tot = NUM_STRINGS, 0
            for string in chord_list.split('.'):
                if string == 'x':
                    played_strings -= 1
                    continue
                else:
                    fret_tot += int(string)
            if played_strings != 0:
                scores.append(fret_tot/played_strings)
            else:
                scores.append(-1)
        return torch.tensor(scores, device=predicted_frets.device)
