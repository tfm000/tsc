from torch import nn
import torch
import torch.nn.functional as F

from mamba.models.blocks import get_block
from mamba.models.encoders import ConvMambaEncoder


class Mamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_classes: int = config.n_classes
        self.d_pred = config.d_pred
        
        self.bayesian_eval: bool = False

        # Residual Connections
        self.residuals: bool = config.residuals
        
        # Mamba encoders
        self.encoders = nn.ModuleList([ConvMambaEncoder(config) 
                                      for _ in range(config.e_layers)])

        # Layer Norm
        if self.residuals:
            self.layer_norm = nn.LayerNorm(config.d_model)
        
        # linear projection
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(config.seq_len * config.d_model, self.n_classes * self.d_pred)
        
    def forward(self, x):
        residuals = x if self.residuals else None

        # mamba encoder
        for encoder in self.encoders:
            encoder.bayesian_eval = self.bayesian_eval
            x = encoder(x)
            if self.residuals:
                x = x + residuals
                x = self.layer_norm(x)
                residuals = x
        
        # linear projection for classification
        batch_size: int = x.shape[0]
        x = self.flatten(x)
        logits = self.linear(x).reshape((batch_size, self.d_pred, self.n_classes))
        pseudo_probs = F.softmax(logits, dim=2)
        return pseudo_probs, logits