from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence

from helpers import RecurrentHelper
from locked_dropout import LockedDropout
from weight_drop import WeightDrop


class RNNModule(nn.Module, RecurrentHelper):
    def __init__(self, ninput,
                 nhidden,
                 rnn_type='LSTM',
                 nlayers=1,
                 bidirectional=False,
                 dropouti=0.,
                 dropoutw=0.,
                 dropouto=0.,
                 dropout=0.,
                 pack=True, last=False):
        """
        A simple RNN Encoder, which produces a fixed vector representation
        for a variable length sequence of feature vectors, using the output
        at the last timestep of the RNN.
        We use batch_first=True for our implementation.
        Tensors are are shape (batch_size, sequence_length, feature_size).
        Args:
            input_size (int): the size of the input features
            rnn_size (int):
            num_layers (int):
            bidirectional (bool):
            dropout (float):
        """
        super(RNNModule, self).__init__()

        self.pack = pack
        self.last = last

        self.lockdrop = LockedDropout()

        assert rnn_type in ['LSTM', 'GRU'], 'RNN type is not supported'

        if not isinstance(nhidden, list):
            nhidden = [nhidden]

        self.rnn_type = rnn_type
        self.ninp = ninput
        self.nhid = nhidden
        self.nlayers = nlayers
        self.dropouti = dropouti               # rnn input dropout
        self.dropoutw = dropoutw               # rnn recurrent dropout
        self.dropouto = dropouto               # rnn output dropout
        if dropout == .0 and dropouto != .0:
            self.dropout = self.dropouto       # rnn output dropout (of the last RNN layer)

        if rnn_type == 'LSTM':
            self.rnns = [nn.LSTM(input_size=ninput if l == 0 else nhidden[l - 1],
                                 hidden_size=nhidden[l],
                                 num_layers=1,
                                 dropout=0,
                                 batch_first=True) for l in range(nlayers)]

            # Dropout to recurrent layers (matrices weight_hh AND weight_ih of each layer of the RNN)
            if dropoutw:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0', 'weight_ih_l0'],
                                        dropout=dropoutw) for rnn in self.rnns]
        # if rnn_type == 'GRU':
        #     self.rnns = [nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
        #     if wdrop:
        #         self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        print(self.rnns)
        self.rnns = nn.ModuleList(self.rnns)

        # self.init_weights()

    def reorder_hidden(self, hidden, order):
        """

        :param hidden:
        :param order:
        :return:
        """
        if isinstance(hidden, tuple):
            hidden = hidden[0][:, order, :], hidden[1][:, order, :]
        else:
            hidden = hidden[:, order, :]

        return hidden

    def init_hidden(self, bsz):
        """
        Initialise the hidden and cell state (h0, c0) for the first timestep (t=0).
        Both h0, c0 are of shape (num_layers * num_directions, batch_size, hidden_size)
        :param bsz: batch size
        :return:
        """
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid[l]).zero_(),
                     weight.new(1, bsz, self.nhid[l]).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]
        # if self.rnn_type == 'LSTM':
        #     return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
        #             weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
        #             for l in range(self.nlayers)]
        # elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
        #     return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
        #             for l in range(self.nlayers)]

    def forward(self, x, hidden=None, lengths=None, return_h=False):
        """

        :param x: tensor of shape (batch_size, seq_len, embedding_size)
        :param hidden: tuple (h0, c0), each of shape (num_layers * num_directions, batch_size, hidden_size)
        :param lengths: tensor (size 1 with true lengths)
        :return:
        """
        batch_size, seq_length, feat_size = x.size()

        # Dropout to inputs of the RNN (dropouti)
        emb = self.lockdrop(x, self.dropouti)

        if hidden is None:
            hidden = self.init_hidden(batch_size)
        # if lengths is not None and self.pack:
        #
        #     ###############################################
        #     # sorting
        #     ###############################################
        #     lenghts_sorted, sorted_i = lengths.sort(descending=True)
        #     _, reverse_i = sorted_i.sort()
        #
        #     x = x[sorted_i]
        #
        #     if hidden is not None:
        #         hidden = self.reorder_hidden(hidden, sorted_i)
        #
        #     ###############################################
        #     # forward
        #     ###############################################
        #     # packed = pack_padded_sequence(x, lenghts_sorted, batch_first=True)
        #     #
        #     # self.rnn.flatten_parameters()
        #     # out_packed, hidden = self.rnn(packed, hidden)
        #     out_packed, hidden = self.rnn(x, hidden)
        #
        #     out_unpacked, _lengths = pad_packed_sequence(out_packed,
        #                                                  batch_first=True,
        #                                                  total_length=max_length)
        #
        #     # out_unpacked = self.dropout(out_unpacked)
        #
        #     ###############################################
        #     # un-sorting
        #     ###############################################
        #     outputs = out_unpacked[reverse_i]
        #     hidden = self.reorder_hidden(hidden, reverse_i)
        #
        # else:
        #     # raise NotImplementedError
        #     # todo: make hidden return the true last states
        #     # self.rnn.flatten_parameters()
        #     outputs, hidden = self.rnn(x, hidden)
        #     # self.rnn.flatten_parameters()
        #     # outputs = self.dropout(outputs)
        #
        # if self.last:
        #     return outputs, hidden, self.last_timestep(outputs, lengths,
        #                                                self.rnn.bidirectional)
        #
        # return outputs, hidden

        raw_output = emb  # input to the first layer
        new_hidden = []
        raw_outputs = []
        outputs = []

        """ 
        `batch_first = True` use of PyTorch RNN module
        shapes of input and output tensors
        -----------------------------------------------
        output, (hn, cn) = rnn(input, (h0, c0))
        -----------------------------------------------
        input: (batch_size, seq_len, input_size)
        h0: (num_layers * num_directions, batch_size, feature_size)
        c0: (num_layers * num_directions, batch_size, feature_size)
        -----------------------------------------------
        output: (batch_size, seq_len, num_directions * hidden_size]) 
        contains the output features `(h_t)` from the last layer of the LSTM, for each `t`
        hn: (num_layers * num_directions, batch_size, feature_size)
        contains the hidden state for `t = seq_len`
        cn: (num_layers * num_directions, batch_size, feature_size)
        contains the cell state for `t = seq_len`
        """

        # for each layer of the RNN
        for l, rnn in enumerate(self.rnns):
            # calculate hidden states and output from the l RNN layer
            raw_output, new_h = rnn(raw_output, hidden[l])
            # save them in lists
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                # apply dropout to the output of the l-th RNN layer (dropouto)
                raw_output = self.lockdrop(raw_output, self.dropouto)
                # save 'dropped-out outputs' in a list
                outputs.append(raw_output)
        hidden = new_hidden

        # Dropout to the output of the last RNN layer (dropout)
        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        # result = output.view(output.size(0) * output.size(1), output.size(2))
        result = output
        # result: output of the last RNN layer
        # hidden: hidden state of the last RNN layer
        # raw_outputs: outputs of all RNN layers without dropout
        # outputs: dropped-out outputs of all RNN layers
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden
