import torch
from torch import nn

from attention_module import SelfAttention
from embed_dropout import EmbeddingDropout
from helpers import RecurrentHelper
from locked_dropout import LockedDropout
from my_rnn_module import RNNModule



class RNNClassifier(nn.Module, RecurrentHelper):
    """
    RNN-based classifier. Contains an embedding layer, a recurrent module,
    a self-attention layer, and an output linear layer for classification.
    """
    def __init__(self, ntokens, nclasses, **kwargs):
        super(RNNClassifier, self).__init__()

        ############################################
        # Params
        ############################################
        self.ntokens = ntokens # vocab size
        self.embed_finetune = kwargs.get("embed_finetune", False)
        self.emb_size = kwargs.get("emb_size", 50)
        self.embed_noise = kwargs.get("embed_noise", .0)

        self.rnn_size = kwargs.get("rnn_size", 100)
        self.rnn_layers = kwargs.get("rnn_layers", 1)

        # dropouts
        self.dropoute = kwargs.get("dropoute", .3) # embedding layer dropout
        self.rnn_dropouti = kwargs.get("rnn_dropouti", .5) # rnn input dropout
        self.rnn_wdrop = kwargs.get("rnn_wdrop", .5) # rnn recurrent dropout
        self.rnn_dropouth = kwargs.get("rnn_dropouth", .5) # rnn output dropout
        self.rnn_dropout = kwargs.get("rnn_dropout", .5) # not sure yet

        self.pack = kwargs.get("pack", True)
        # self.no_rnn = kwargs.get("no_rnn", False)
        self.bidir = kwargs.get("bidir", False)
        self.attention = kwargs.get("attention", False)

        if not isinstance(self.rnn_size, list):
            self.rnn_size = [self.rnn_size]

        ############################################
        # Layers
        ############################################
        self.embedding = EmbeddingDropout(num_embeddings=ntokens,
                                          embedding_dim=self.emb_size,
                                          embedding_dropout=self.dropoute)


        self.rnn = RNNModule(ninput=self.emb_size,
                             nhidden=self.rnn_size,
                             nlayers=self.rnn_layers,
                             dropouti=self.rnn_dropouti,
                             dropouth=self.rnn_dropouth,
                             dropout=self.rnn_dropout,
                             wdrop=self.rnn_wdrop)

        if self.attention:
            self.attention_size = self.rnn_size[-1]

            self.attention = SelfAttention(self.attention_size, baseline=True)

            self.classes = nn.Linear(self.attention_size, nclasses)
        else:
            self.classes = nn.Linear(self.rnn_size[-1], nclasses)

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

        # # this applies dropout to DIMENSIONS of the word embedding vectors
        # # I don't know why or why we want it....
        # emb = self.lockdrop(embeds, self.rnn_dropouti)

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
            representations = outputs
            attentions = []
        ############################################
        # Output classification layer
        ############################################
        logits = self.classes(representations)

        return logits, representations, attentions
