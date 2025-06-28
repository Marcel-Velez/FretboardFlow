import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim


class BiLSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def training_step(self, batch, batch_idx):
        chords, midi_prev, midi_target = batch
        output = self(chords, midi_prev)
        loss = nn.MSELoss()(output, midi_target)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
