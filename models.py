import torch
from torch import nn

from embed_dropout import EmbeddingDropout
from helpers import RecurrentHelper
from locked_dropout import LockedDropout
from rnn_model import RNNModel


class RNNClassifier(nn.Module, RecurrentHelper):
    def __init__(self, ntokens, nclasses, **kwargs):
        super(RNNClassifier, self).__init__()

        ############################################
        # Params
        ############################################
        self.ntokens = ntokens
        self.embed_finetune = kwargs.get("embed_finetune", False)
        self.emb_size = kwargs.get("emb_size", 50)
        self.embed_noise = kwargs.get("embed_noise", .0)
        self.dropoute = kwargs.get("dropoute", .7)
        self.rnn_size = kwargs.get("rnn_size", 100)
        self.rnn_layers = kwargs.get("rnn_layers", 2)
        self.rnn_dropouti = kwargs.get("rnn_dropouti", .5)
        self.rnn_dropouto = kwargs.get("rnn_dropouto", .5)
        self.rnn_dropoutw = kwargs.get("rnn_dropoutw", .5)
        self.pack = kwargs.get("pack", True)
        self.no_rnn = kwargs.get("no_rnn", False)
        self.bidir = kwargs.get("bidir", False)

        self.lockdrop = LockedDropout()  # den kserw ti einai auto
        self.idrop = nn.Dropout(self.rnn_dropouti)
        self.hdrop = nn.Dropout(self.rnn_dropoutw)
        dropout=0.5
        self.drop = nn.Dropout(dropout)
        ############################################
        # Layers
        ############################################
        self.embedding = EmbeddingDropout(num_embeddings=ntokens,
                                          embedding_dim=self.emb_size,
                                          embedding_dropout=self.dropoute)

        self.rnn = RNNModel(input_size=self.emb_size,
                            rnn_size=self.rnn_size,
                            rnn_layers=self.rnn_layers,
                            bidirectional=self.bidir,
                            pack=self.pack)

        if self.no_rnn == False:
            self.attention_size = self.rnn_size
        else:
            self.attention_size = self.emb_size

        # self.attention = SelfAttention(self.attention_size, baseline=True)

        self.classes = nn.Linear(self.attention_size, nclasses)

    def initialize_embeddings(self, embs, trainable=False):

        freeze = not trainable

        embeddings = torch.from_numpy(embs).float()
        embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze)

        self.word_embedding = embedding_layer

    def forward(self, src, lengths=None):
        ############################################
        # Embed input sequences
        ############################################
        embeds = self.embedding(src)
        emb = self.lockdrop(embeds, self.rnn_dropouti)
        if self.no_rnn == False:
            # step 2: encode the sentences
            outputs, _ = self.rnn(embeds, lengths=lengths)
        else:
            outputs = embeds

        representations, attentions = self.attention(outputs, lengths)

        # step 3: output layers
        logits = self.classes(representations)

        return logits, representations, attentions
