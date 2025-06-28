import torch
import torch.nn as nn

from .config import NUM_STRINGS, NUM_FRETS, PITCH_CLASSES
from .base_ff_model import BaseGuitarTrainer


# ----------------------------------------------------------------------------------------------------------------------
# https://github.com/Maghoumi/DeepGRU/blob/master/model.py only diference is the adjustable hidden_size and converting
# it to a pytorch lightning module
class DeepGRU(BaseGuitarTrainer):
    def __init__(self,
                 chord_input_size=36,
                 hidden_size=128,
                 num_layers=1,
                 learning_rate=1e-3,
                 uncertainty_weight=0.5,
                 fretboardflow=True, **kwargs):
        super().__init__(
            hidden_size=hidden_size,
            num_layers=num_layers,
            uncertainty_weight=uncertainty_weight,
            learning_rate=learning_rate,
            fretboardflow=fretboardflow,
            **kwargs
        )
        # self.num_features = input_size
        self.learning_rate = learning_rate

        hidden_size =128
        # Encoder
        self.gru1 = nn.GRU(chord_input_size, hidden_size, 2, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, 2, batch_first=True)
        self.gru3 = nn.GRU(hidden_size//2, hidden_size//4, 1, batch_first=True)

        # Encoder
        self.chordgru1 = nn.GRU(NUM_STRINGS * (NUM_FRETS+1), hidden_size, 2, batch_first=True)
        self.chordgru2 = nn.GRU(hidden_size, hidden_size//2, 2, batch_first=True)
        self.chordgru3 = nn.GRU(hidden_size//2, hidden_size//4, 1, batch_first=True)

        # Attention
        self.attention = Attention(hidden_size//2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size//2),
            nn.Dropout(0.5),
            nn.Linear(hidden_size//2, NUM_STRINGS*(NUM_FRETS+1))
        )

    def forward(self, chords_history, frets_prev_one_hot):
        batch_size = chords_history.shape[0]
        # Encode
        output, _ = self.gru1(chords_history)
        output, _ = self.gru2(output)
        output_chord_history, history_hidden = self.gru3(output)


        if len(frets_prev_one_hot.shape) == 4:
            # e.g. (batch, hist_len, 6, 25)
            b, h, s, f = frets_prev_one_hot.shape
            position_in = frets_prev_one_hot.reshape(b, s, -1)
        else:
            position_in = frets_prev_one_hot


        # Encode
        output, _ = self.chordgru1(position_in)
        output, _ = self.chordgru2(output)
        position_output, position_hidden = self.chordgru3(output)


        combined_output = torch.cat([position_output, output_chord_history[:,1:,:]], dim=-1)
        combined_history = torch.cat([history_hidden, position_hidden], dim=-1)

        # Pass to attention with the original padding
        attn_output = self.attention(combined_output, combined_history)

        # Classify
        classify = self.classifier(attn_output)
        return classify.reshape(batch_size, NUM_STRINGS, -1)

    def forward_single(self, x, x_lengths):

        output, _ = self.gru1(x)
        output, _ = self.gru2(output)
        output, hidden = self.gru3(output)
        attn_output = self.attention(output, hidden[-1:])
        classify = self.classifier(attn_output)
        return classify


# ----------------------------------------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, attention_dim):
        super(Attention, self).__init__()
        self.w = nn.Linear(attention_dim, attention_dim, bias=False)
        self.gru = nn.GRU(attention_dim, attention_dim, 1, batch_first=True)

    def forward(self, input_padded, hidden):
        # e = torch.bmm(self.w(input_padded), hidden.permute(1, 2, 0))
        test_a = self.w(input_padded)
        test_b = hidden.permute(1, 2, 0)
        e = torch.bmm(test_a, test_b)
        context = torch.bmm(input_padded.permute(0, 2, 1), e.softmax(dim=1)) # the attention is after softmax
        context = context.permute(0, 2, 1)

        # Compute the auxiliary context, and concat
        aux_context, _ = self.gru(context, hidden)
        output = torch.cat([aux_context, context], 2).squeeze(1)

        return output



