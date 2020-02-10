import numpy as np
import torch

from model import RNNClassifier

"""
This is the main script to do a forward pass of the Variational RNN model.
"""

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch:", torch.__version__)
    print("Cuda:", torch.backends.cudnn.cuda)
    print("CuDNN:", torch.backends.cudnn.version())
    print('device: {}'.format(device))


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