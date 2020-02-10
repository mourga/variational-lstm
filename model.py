import torch
from torch import nn

from attention_module import SelfAttention
from embed_dropout import EmbeddingDropout
from helpers import RecurrentHelper
from rnn_module import RNNModule


class RNNClassifier(nn.Module, RecurrentHelper):
    """
    RNN-based classifier. Contains an embedding layer, a recurrent module,
    a self-attention layer (optional), and an output linear layer for classification.
    """

    def __init__(self, ntokens, nclasses, **kwargs):
        super(RNNClassifier, self).__init__()

        ############################################
        # Params
        ############################################
        self.ntokens = ntokens    # vocab size
        self.nclasses = nclasses  # number of classes

        # embedding layer
        self.embed_finetune = kwargs.get("embed_finetune", False)
        self.emb_size = kwargs.get("emb_size", 50)
        self.embed_noise = kwargs.get("embed_noise", .0)

        # RNN encoder
        self.rnn_size = kwargs.get("rnn_size", 100)
        self.rnn_layers = kwargs.get("rnn_layers", 1)
        self.bidir = kwargs.get("bidir", False)

        # dropouts
        self.dropoute = kwargs.get("dropoute", .0)          # embedding layer dropout
        self.rnn_dropouti = kwargs.get("rnn_dropouti", .0)  # rnn input dropout
        self.rnn_dropoutw = kwargs.get("rnn_dropoutw", .0)  # rnn recurrent dropout
        self.rnn_dropouto = kwargs.get("rnn_dropouto", .0)  # rnn output dropout
        self.rnn_dropout = kwargs.get("rnn_dropout", .0)    # rnn output dropout (of last RNN layer)

        self.pack = kwargs.get("pack", True)
        # self.no_rnn = kwargs.get("no_rnn", False)

        self.attention = kwargs.get("attention", False)

        if not isinstance(self.rnn_size, list):
            self.rnn_size = [self.rnn_size]

        ############################################
        # Layers
        ############################################
        self.embedding = EmbeddingDropout(num_embeddings=self.ntokens,
                                          embedding_dim=self.emb_size,
                                          embedding_dropout=self.dropoute)

        self.rnn = RNNModule(ninput=self.emb_size,
                             nhidden=self.rnn_size,
                             nlayers=self.rnn_layers,
                             dropouti=self.rnn_dropouti,
                             dropoutw=self.rnn_dropoutw,
                             dropouto=self.rnn_dropouto)

        if self.attention:
            self.attention_size = self.rnn_size[-1]
            self.attention = SelfAttention(self.attention_size, baseline=True)
            self.classes = nn.Linear(self.attention_size, self.nclasses)
        else:
            self.classes = nn.Linear(self.rnn_size[-1], self.nclasses)

    def initialize_embeddings(self, embs, trainable=False):
        """
        Initialise embedding layer with pre-trained word embeddings.
        :param embs:
        :param trainable: if True, they are trained, if False they remain frozen.
        :return:
        """

        freeze = not trainable

        embeddings = torch.from_numpy(embs).float()
        embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze)

        self.embedding = embedding_layer

    def forward(self, src, lengths=None):
        ############################################
        # Embed input sequences
        ############################################
        embeds = self.embedding(src)

        ############################################
        # Encode sequences
        ############################################
        outputs, _ = self.rnn(embeds, lengths=lengths)

        if self.attention:
            ########################################
            # Apply self-attention
            ########################################
            representations, attentions = self.attention(outputs, lengths)
        else:
            ########################################
            # Use last RNN hidden state
            ########################################
            representations = outputs[:,-1,:]
            attentions = []

        ############################################
        # Output classification layer
        ############################################
        logits = self.classes(representations)

        return logits, representations, attentions
