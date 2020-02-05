import torch
import torch.nn as nn

from embed_dropout import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

"""
Code from https://github.com/salesforce/awd-lstm-lm
paper: https://arxiv.org/pdf/1708.02182.pdf (see Section )
"""


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
                 input_size,
                 rnn_size,
                 rnn_layers=1,
                 bidirectional=False,
                 pack=True,
                 rnn_type='LSTM', dropout=0.5,
                 dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):

        super(RNNModel, self).__init__()

        self.lockdrop = LockedDropout()  # den kserw ti einai auto
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        # self.encoder = nn.Embedding(ntoken, ninp)

        self.rnn_type = rnn_type
        self.ninp = input_size
        self.nhid = rnn_size
        self.nlayers = rnn_layers
        self.dropout = dropout
        self.dropouti = dropouti  # input
        self.dropouth = dropouth  # hidden
        self.dropoute = dropoute  # embedding
        self.tie_weights = tie_weights

        assert rnn_type in ['LSTM', 'GRU'], 'RNN type is not supported'

        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(self.ninp if l == 0 else self.nhid, self.nhid if l != self.nlayers - 1
            else (self.ninp if tie_weights else self.nhid), 1, dropout=0)
                         for l in range(self.nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(self.ninp if l == 0 else self.nhid, self.nhid if l != self.nlayers - 1
            else self.ninp, 1, dropout=0) for l
                         in range(self.nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]

        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.init_weights()

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        # initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                param.data[self.nhid:2 * self.nhid] = 1

    def forward(self, input, hidden, return_h=False, lengths=None):
        # emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)


        # emb = self.lockdrop(emb, self.dropouti)

        raw_output = input
        new_hidden = []
        # raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                # self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0) * output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                self.ninp if self.tie_weights else self.nhid)).zero_(),
                     weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                         self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]
