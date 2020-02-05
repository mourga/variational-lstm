import numpy as np
import torch

from models import RNNClassifier

if __name__ == '__main__':
    V = 1000  # vocabulary size
    h = 4  # embedding size
    bptt = 10  # sequence length
    batch_size = 2  # batch size
    emb_drop = 0.1  # dropout to be applied to the embedding layer

    # dummy input sequence
    # words = np.random.random_integers(low=0, high=V - 1, size=(batch_size, bptt))
    words = np.array([[2,3,4,5,6,7,2,2], [245,76,134,2,456,23,7,6]])

    words = torch.LongTensor(words)

    # # embedding layer
    # embed = torch.nn.Embedding(V, h)
    #
    # # without embedding dropout
    # origX = embed(words)
    #
    # # wit embedding dropout
    # X = embedded_dropout(embed, words, emb_drop)

    model = RNNClassifier(V,3)

    print(model)
    model(words)