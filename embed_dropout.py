import numpy as np
import torch
from torch import nn

"""
Code from https://github.com/salesforce/awd-lstm-lm
paper: https://arxiv.org/pdf/1708.02182.pdf (see Section 4.3)
"""


class EmbeddingDropout(nn.Module):
    """
    Embedding Layer. if dropout != 0. we apply dropout to word 'types' not 'tokens' as suggested
    in the paper https://arxiv.org/pdf/1512.05287.pdf.
    We first map the input sequences to the corresponding embeddings (from |V| -> embedding_dim) and THEN apply dropout.
    """
    def __init__(self, num_embeddings, embedding_dim, embedding_dropout=0.):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dropoute = embedding_dropout

        self.embed = nn.Embedding(num_embeddings=self.num_embeddings,
                                  embedding_dim=self.embedding_dim)

    def forward(self, words):
        if self.dropoute and self.training:
            mask = self.embed.weight.data.new().resize_((self.embed.weight.size(0), 1)).bernoulli_(1 - self.dropoute).expand_as(
                self.embed.weight) / (1 - self.dropoute)
            masked_embed_weight = mask * self.embed.weight
        else:
            masked_embed_weight = self.embed.weight

        padding_idx = self.embed.padding_idx  # be careful here to use the same 'padding_idx' name
        if padding_idx is None:
            padding_idx = -1

        X = torch.nn.functional.embedding(words, masked_embed_weight,
                                          padding_idx, self.embed.max_norm, self.embed.norm_type,
                                          self.embed.scale_grad_by_freq, self.embed.sparse
                                          )
        return X


def embedded_dropout(embed, words, dropout=0.1):
    """
    Embedding layer dropout.
    :param embed: embedding layer
    :param words: input sequence of words. shape: (batch size x sequence length)
    :param dropout: dropout to be applied to the embedding layer
    :return:
    """
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight

    padding_idx = embed.padding_idx # be careful here to use the same 'padding_idx' name
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight,
                                      padding_idx, embed.max_norm, embed.norm_type,
                                      embed.scale_grad_by_freq, embed.sparse
                                      )
    return X


if __name__ == '__main__':
    V = 50  # vocabulary size
    h = 4  # embedding size
    bptt = 10  # sequence length
    batch_size = 2  # batch size
    emb_drop = 0.1  # dropout to be applied to the embedding layer

    # dummy input sequence
    words = np.random.random_integers(low=0, high=V - 1, size=(batch_size, bptt))
    words = torch.LongTensor(words)

    # embedding layer
    embed = torch.nn.Embedding(V, h)

    # without embedding dropout
    origX = embed(words)

    # wit embedding dropout
    X = embedded_dropout(embed, words, emb_drop)
