import torch
import torch.nn as nn

from mamba.models.blocks import get_block



class BiMambaEncoderBase(nn.Module):
    def __init__(self, block_config, block: str, dropout: float):
        super(BiMambaEncoderBase, self).__init__()

        # forward layers
        block_obj = get_block(block=block)
        self.block_fwd = block_obj(block_config)
        self.layer_norm_fwd = nn.LayerNorm(block_config.d_model)

        # backward layers
        self.block_bwd = block_obj(block_config)
        self.layer_norm_bwd = nn.LayerNorm(block_config.d_model)

        # common layers
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        # forward direction
        x_fwd = self.block_fwd(x)
        x_fwd = self.dropout(x_fwd)
        x_fwd = self.layer_norm_fwd(x + x_fwd)
        
        # backward direction
        x_bwd = torch.flip(x, dims=[1])
        x_bwd = self.block_bwd(x_bwd)
        x_bwd = torch.flip(x_bwd, dims=[1])
        x_bwd = self.dropout(x_bwd)
        x_bwd = self.layer_norm_bwd(x + x_bwd)

        return x_fwd, x_bwd


class BiMamba4TSEncoder(BiMambaEncoderBase):
    def __init__(self, block_config, block: str, dropout: float, d_hidden: int):
        super(BiMamba4TSEncoder, self).__init__(block_config=block_config, 
                                         block=block, dropout=dropout)

        # forward layers
        self.ff_fwd = nn.Sequential(
            nn.Linear(block_config.d_model, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, block_config.d_model)
        )
        self.layer_norm_fwd2 = nn.LayerNorm(block_config.d_model)

        # backward layers
        self.ff_bwd = nn.Sequential(
            nn.Linear(block_config.d_model, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, block_config.d_model)
        )
        self.layer_norm_bwd2 = nn.LayerNorm(block_config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # getting halfway point
        x_resid_fwd, x_resid_bwd = super().forward(x)

        # continuing forward direction
        x_fwd = self.ff_fwd(x_resid_fwd)
        x_fwd = self.dropout(x_fwd)
        x_fwd = self.layer_norm_fwd2(x_fwd + x_resid_fwd)

        # continuing backward direction
        x_bwd = self.ff_bwd(x_resid_bwd)
        x_bwd = self.dropout(x_bwd)
        x_bwd = self.layer_norm_bwd2(x_bwd + x_resid_bwd)

        return x_fwd + x_bwd
        

class BiMambaPlusEncoder(BiMambaEncoderBase):
    def __init__(self, block_config, block: str, dropout: float, d_hidden: int):
        super(BiMambaPlusEncoder, self).__init__(block_config=block_config, 
                                                 block=block, dropout=dropout)
        # common layers
        self.ff = nn.Sequential(
            nn.Linear(block_config.d_model, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, block_config.d_model),
        )
        self.layer_norm = nn.LayerNorm(block_config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # getting halfway point
        x_fwd, x_bwd = super().forward(x)

        # continuing joint direction
        x_resid = x_fwd + x_bwd
        x = self.ff(x_resid)
        x = self.dropout(x)
        x = self.layer_norm(x + x_resid)
        return x


class BiMambaFusionEncoder(BiMambaEncoderBase):
    """Identical to Bi-Mamba+, only here we concatinate / fuse the forward and 
    backward branches, rather than adding them."""

    def __init__(self, block_config, block: str, dropout: float, d_hidden: int):
        super(BiMambaFusionEncoder, self).__init__(block_config=block_config, 
                                             block=block, dropout=dropout)
        # common layers
        self.linear = nn.Linear(block_config.d_model * 2, block_config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(block_config.d_model, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, block_config.d_model),
        )
        self.layer_norm = nn.LayerNorm(block_config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # getting halfway point
        x_fwd, x_bwd = super().forward(x)

        # continuing joint direction
        x_cat = torch.cat([x_fwd, x_bwd], dim=len(x_fwd.shape)-1) # concat on vars dim
        x_resid = self.linear(x_cat)
        x = self.ff(x_resid)
        x = self.dropout(x)
        x = self.layer_norm(x + x_resid)
        return x


class ConvMambaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Mamba block
        block_obj = get_block(block=config.block_name)
        self.block = block_obj(config)
        self.layer_norm = nn.LayerNorm(config.emb_size)

        # Feed forward net
        self.ff = nn.Sequential(
            nn.Linear(config.emb_size, config.dim_ff),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_ff, config.emb_size),
            nn.Dropout(config.dropout),
            )

        self.dropout = nn.Dropout(p=config.dropout)
        self.bayesian_eval: bool = False

    def forward(self, x):
        x_resid = x
        x = self.block(x)
        if self.bayesian_eval:
            x = self.dropout(x)
        x_resid = self.layer_norm(x + x_resid)
        x = self.ff(x_resid)
        x = self.layer_norm(x + x_resid)
        return x