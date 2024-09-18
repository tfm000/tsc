import torch
import torch.nn as nn
import torch.nn.functional as F
from itransformer.layers.Transformer_EncDec import Encoder, EncoderLayer
from itransformer.layers.SelfAttention_Family import FullAttention, AttentionLayer
from itransformer.layers.Embed import DataEmbedding_inverted

import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, config):
        super(Model, self).__init__()
        self.n_classes: int = config.n_classes
        self.seq_len = config.seq_len
        self.use_norm = config.use_norm
        self.d_pred = config.d_pred
        self.bayesian_eval = False

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(config.seq_len, config.emb_size, config.dropout)

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 5, attention_dropout=config.dropout,
                                      output_attention=False), config.emb_size, 
                                      config.num_heads),
                    config.emb_size,
                    config.dim_ff,
                    dropout=config.dropout,
                    activation=config.activation
                ) for l in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.emb_size)
        )

        # linear projection head
        self.flatten = nn.Flatten()
        self.projector = nn.Linear(config.emb_size * config.d_model, self.n_classes * self.d_pred, bias=True)

    def forward(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        # enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens

        ########## Tyler Comment
        ## x_mark_enc is the positional embedding. in the energy dataset, he 
        # uses a time specific embedding -> may also be useful for FX/Finance 
        # data, as this can be influenced by market openings etc which may be 
        # easier to learn than distance based embedding methods.
        ##########
        enc_out = self.enc_embedding(x_enc)

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = self.flatten(enc_out)
        batch_size: int = x_enc.shape[0]
        logits = self.projector(enc_out).reshape((batch_size, self.d_pred, self.n_classes))
        pseudo_probs = F.softmax(logits, dim=2)
        return pseudo_probs, logits
        