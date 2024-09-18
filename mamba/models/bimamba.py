import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba.models.revin import RevIN
from mamba.models.encoders import BiMamba4TSEncoder, BiMambaPlusEncoder, BiMambaFusionEncoder


class BiMamba(nn.Module):
    def __init__(self, config):
        
        super(BiMamba, self).__init__()

        self.n_classes: int = config.n_classes
        self.d_pred: int = config.d_pred

        # instance norm and denorm
        self.revin: RevIN = RevIN(num_features=config.d_model)

        # creates J patches of length P and stride S from an input
        if config.patching is None:
            self.patching = (lambda x: x)
            # implement an else / elif later - will use the config arg

        if config.encoder_name == 'bimamba+':
            encoder_obj = BiMambaPlusEncoder
        elif config.encoder_name == 'bimamba4ts':
            encoder_obj = BiMamba4TSEncoder
        elif config.encoder_name == 'bimambafusion':
            encoder_obj = BiMambaFusionEncoder
        else:
            raise ValueError(f"{config.encoder_name} is not a valid encoder.")

        self.encoders = nn.ModuleList([
            encoder_obj(block_config=config, block=config.block_name, 
                        dropout=config.dropout, d_hidden=config.emb_size)
                        for _ in range(config.e_layers)])
        
        # linear projection
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(config.seq_len * config.d_model, self.n_classes * self.d_pred)
        
    def forward(self, x: torch.Tensor) -> torch.tensor:
        # instance norm
        x = self.revin(x, 'norm')

        # patching / tokenisation
        x = self.patching(x)

        # passing through encoder
        for encoder in self.encoders:
            x = encoder(x)
        
        # linear projection for classification
        batch_size: int = x.shape[0]
        x = self.flatten(x)
        logits = self.linear(x).reshape((batch_size, self.d_pred, self.n_classes))
        pseudo_probs = F.softmax(logits, dim=2)
        return pseudo_probs, logits


        
