import numpy as np
import torch

from model import RNNClassifier

"""
This is the main script to do a forward pass of the Variational RNN model.
"""

if __name__ == '__main__':
    # dummy example
    V = 1000          # vocabulary size
    emb_size = 5      # embedding size
    bptt = 4          # sequence length
    batch_size = 2    # batch size

    # choose dropouts
    dropoute = 0.2    # dropout to the embedding layer
    dropouti = 0.2    # dropout to the inputs of the RNN
    dropouto = 0.3    # dropout to the outputs of the RNN
    dropoutw = 0.4    # dropout to the recurrent layers of the RNN

    # dummy input tensor of shape (batch_size, seq_len)
    words = np.random.random_integers(low=0, high=V - 1, size=(batch_size, bptt))
    # words = np.array([[2,3,4,5,6,7,2,2], [245,76,134,2,456,23,7,6]])

    words = torch.LongTensor(words)

    model = RNNClassifier(ntokens=V, nclasses=3,
                          emb_size=emb_size,
                          dropoute=dropoute,
                          rnn_size=[6,7,8],
                          rnn_layers=3,
                          rnn_dropouti=dropouti,
                          rnn_dropouto=dropouto,
                          rnn_dropoutw=dropoutw)

    print(model)

    # forward pass
    model(words)