from torch import nn
import torch
import torch.nn.functional as F

from mamba.models.blocks import get_block
from mamba.models.encoders import ConvMambaEncoder


class ConvMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_pred: int = config.d_pred
        self.n_classes: int = config.n_classes
        self.residuals: bool = config.residuals

        self.dropout = nn.Dropout(p=config.dropout)
        self.bayesian_eval: bool = False

        # Embedding Layers -----------------------------------------------------------
        self.embed_layer = nn.Sequential(
            nn.Conv2d(1, config.emb_size*config.channel_expand, kernel_size=[1, 8], padding='same'), 
            nn.BatchNorm2d(config.emb_size*config.channel_expand),
            nn.GELU())

        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(config.emb_size*config.channel_expand, config.emb_size, kernel_size=[config.nvars, 1], padding='valid'),
            nn.BatchNorm2d(config.emb_size),
            nn.GELU())
        
        # Mamba encoders
        self.encoders = nn.ModuleList([ConvMambaEncoder(config) 
                                      for _ in range(config.e_layers)])

        # Layer Norm
        self.layer_norm = nn.LayerNorm(config.emb_size)
        
        # linear projection
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(config.seq_len * config.emb_size, self.n_classes * self.d_pred)

    def forward(self, x):
        # convtran embedding
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        if self.bayesian_eval:
            x_src = self.dropout(x_src)
        x_src = self.embed_layer2(x_src).squeeze(2)

        # mamba encoder
        x_enc = x_src.permute(0, 2, 1)
        residuals = x_enc if self.residuals else None
        for encoder in self.encoders:
            encoder.bayesian_eval = self.bayesian_eval
            x_enc = encoder(x_enc)
            x_enc = x_enc + residuals if self.residuals else x_enc
            x_enc = self.layer_norm(x_enc)
            residuals = x_enc if self.residuals else None

        # linear projection for classification
        batch_size = x_enc.shape[0]
        x_enc = self.flatten(x_enc)
        logits = self.linear(x_enc).reshape((batch_size, self.d_pred, self.n_classes))
        pseudo_probs = F.softmax(logits, dim=2)
        return pseudo_probs, logits