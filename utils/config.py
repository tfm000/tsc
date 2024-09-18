"""Contains configuration objects to pass to the respective models when 
training. Utilising such objects allows for easy tracking of progress when 
performing hyperparameter sensitivity analysis and fitting. Each of the below 
config classes have default values as specified either by their official code 
implementations, or as specified in their original papers. As many of the models
implemented were originally designed for language modelling, the default model d
imensions will likely be too large for financial TSF/TSC tasks. However, they do 
provide an indication on which proportions to target.

Notes
-----
When editing the underlying classes, these can be quite sensitive. Note that 
when inheriting, type hints matter and must be specified each time.
The following stackoverflow thread also contains a lot of useful information on 
dataclasses, if stuck: 
https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses


Implemented
-----------
Daily FX data configuration:
    - ConvTran
    - iTransformer
    - Bi-Mamba
    - Bi-Mamba Fusion
    - ConvMamba
    - Mamba
"""
from dataclasses import dataclass
from convtran.Models.optimizers import get_optimizer
import math
import torch
from typing import Callable

################################################################################
# Dataset specific values
################################################################################
# Daily FX dataset
# fx_d_model=10
# fx_d_pred=10
# fx_seq_len=96
# fx_n_classes=3
ufx_d_model=1
ufx_d_pred=1
ufx_seq_len=96
ufx_n_classes=3

mfx_d_model=4
mfx_d_pred=1
mfx_seq_len=96
mfx_n_classes=3

mfx2_d_model=1
mfx2_d_pred=1
mfx2_seq_len=96
mfx2_n_classes=3

mvfx_d_model=5
mvfx_d_pred=1
mvfx_seq_len=96
mvfx_n_classes=3
################################################################################
# Base config
################################################################################

_no_default = object()

@dataclass
class BaseConfig:
    """Base configuration class to be inherited."""
    name: str = 'Base'

    # varies by sequence
    d_model: int = _no_default # number of variables
    seq_len: int = _no_default # length of training sequence
    n_classes: int = _no_default # number of predition classes
    emb_size: int = _no_default
    d_pred: int = _no_default # number of variables to predict
    dataset_name: str = _no_default

    # optional model spec
    dropout: float = 0

    # optimiser settings
    lr: float = _no_default
    weight_decay: float = 0
    optimiser: str | Callable = _no_default
    
    def __post_init__(self):
        if self.d_pred is _no_default:
            self.d_pred = self.d_model

        # dataset default parameters
        if self.dataset_name == 'univariate_fx_daily':
            if self.d_model is _no_default:
                self.d_model = ufx_d_model
            if self.seq_len is _no_default: 
                self.seq_len = ufx_seq_len
            if self.n_classes is _no_default:
                self.n_classes = ufx_n_classes
            if self.d_pred is _no_default:
                self.d_pred = ufx_d_pred

        elif self.dataset_name == 'momentum_fx_daily':
            if self.d_model is _no_default:
                self.d_model = mfx_d_model
            if self.seq_len is _no_default: 
                self.seq_len = mfx_seq_len
            if self.n_classes is _no_default:
                self.n_classes = mfx_n_classes
            if self.d_pred is _no_default:
                self.d_pred = mfx_d_pred

        elif self.dataset_name == 'momentum_fx_daily_model2':
            if self.d_model is _no_default:
                self.d_model = mfx2_d_model
            if self.seq_len is _no_default: 
                self.seq_len = mfx2_seq_len
            if self.n_classes is _no_default:
                self.n_classes = mfx2_n_classes
            if self.d_pred is _no_default:
                self.d_pred = mfx2_d_pred


        elif self.dataset_name == 'multivariate_fx_daily':
            if self.d_model is _no_default:
                self.d_model = mvfx_d_model
            if self.seq_len is _no_default: 
                self.seq_len = mvfx_seq_len
            if self.n_classes is _no_default:
                self.n_classes = mvfx_n_classes
            if self.d_pred is _no_default:
                self.d_pred = mvfx_d_pred

        # elif self.dataset_name == 'multivariate_fx_daily_model2':
        #     if self.d_model is _no_default:
        #         self.d_model = mfx2_d_model
        #     if self.seq_len is _no_default: 
        #         self.seq_len = mfx2_seq_len
        #     if self.n_classes is _no_default:
        #         self.n_classes = mfx2_n_classes
        #     if self.d_pred is _no_default:
        #         self.d_pred = mfx2_d_pred

        if isinstance(self.optimiser, str):
            self.optimiser = get_optimizer(self.optimiser)
        if not isinstance(self.optimiser, Callable):
            raise TypeError("optmiser must be a string or callable function.")
        

################################################################################
# ConvTran
################################################################################
@dataclass
class ConvTranConfig(BaseConfig):
    """Configuration class for the ConvTran model. Default hyperparameters are 
    as given in the official implementation, which assumes a sequence length 
    (seq_len) of 40 (for the segmentation dataset example).
    https://github.com/Navidfoumani/ConvTran/blob/main/main.py
    https://github.com/Navidfoumani/ConvTran/blob/main/Dataset/load_segment_data.py
    """
    # optional model spec
    emb_size: int = 16 # internal dimension of transformer embeddings
    dim_ff: int = 256 # dimension of dense feedforward part of transformer layer
    num_heads: int = 8 # number of multi-headed attention heads
    fix_pos_encode: str = 'tAPE' # fixed position embedding
    rel_pos_encode: str = 'eRPE' # relative position embedding
    # dropout: float = 0.01
    dropout=0.2
    kernel_sizes: tuple = (1, 8)
    channel_expand: int = 4
    
    # optimiser settings
    optimiser: str | Callable = 'RAdam'
    # lr: float = 1e-3
    lr: float = 1e-4


# Objects
convTranDefault: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    )


convTran_100ff: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=100,
    )

convTran_50ff: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=50,
    )

convTran_25ff: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=25,
    )


convTran_16ff: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=16,
    )

convTran_10ff: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=10,
    )

convTran_8ff: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=8,
    )

convTran_6ff: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=6,
    )

convTran_4ff: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=4,
    )

convTran_3ff: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=3,
    )

convTran_2ff: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    )

convTran_1ff: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=1,
    )
    

convTran_3ff_3channel: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=3,
    channel_expand=3,
    )

convTran_3ff_2channel: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=3,
    channel_expand=2,
    )

convTran_3ff_1channel: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=3,
    channel_expand=1,
    )

convTran_2ff_3channel: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=3,
    )

convTran_2ff_2channel: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    )

convTran_2ff_1channel: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=1,
    )

convTran_2ff_2channel_8emb: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    )


convTran_3ff_2channel_8emb: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=3,
    channel_expand=2,
    emb_size=8,
    )


convTran_2ff_2channel_4head: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    num_heads=4,
    )

convTran_2ff_2channel_3head: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    num_heads=3,
    )

convTran_2ff_2channel_2head: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    num_heads=2,
    )

convTran_2ff_2channel_1head: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    num_heads=1,
    )

# seq len study --- new
convTran_2ff_2channel_8emb_120len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=120,
    )
    
convTran_2ff_2channel_8emb_60len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=60,
    )

convTran_2ff_2channel_8emb_50len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=50,
    )

convTran_2ff_2channel_8emb_40len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=40,
    )

convTran_2ff_2channel_8emb_30len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=30,

    # lr=1e-2,
    # lr=2.5*1e-2,
    # lr=5*1e-2,
    lr=7.5*1e-2,

    # lr=1e-3,
    # lr=7.5*1e-4,
    # lr=5*1e-4,
    # lr=2.5*1e-4,
    
    # lr=7.5*1e-5,
    # lr=5*1e-5,

    # dropout=0.5,
    # dropout=0.4,
    # dropout=0.3,
    # dropout=0.25,

    # dropout=0.15,
    # dropout=0.1,
    # dropout=0.05,
    # dropout=0,
    )

convTran_2ff_2channel_8emb_25len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=25,
    )

convTran_2ff_2channel_8emb_20len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=20,
    )

convTran_2ff_2channel_8emb_15len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=15,
    )

convTran_2ff_2channel_8emb_10len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=10,
    )

convTran_2ff_2channel_8emb_5len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=5,
    )


# seq len study
convTran_3ff_120len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    lr=1e-4,
    dim_ff=3,
    dropout=0.2,
    seq_len=120,
    )

convTran_3ff_60len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    lr=1e-4,
    dim_ff=3,
    dropout=0.2,
    seq_len=60,
    )

convTran_3ff_50len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    lr=1e-4,
    dim_ff=3,
    dropout=0.2,
    seq_len=50,
    )

convTran_3ff_40len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    lr=1e-4,
    dim_ff=3,
    dropout=0.2,
    seq_len=40,
    )

convTran_3ff_30len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    lr=1e-4,
    dim_ff=3,
    dropout=0.2,
    seq_len=30,
    )

convTran_3ff_20len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    lr=1e-4,
    dim_ff=3,
    dropout=0.2,
    seq_len=20,
    )

convTran_3ff_15len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    lr=1e-4,
    dim_ff=3,
    dropout=0.2,
    seq_len=15,
    )

convTran_3ff_10len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    lr=1e-4,
    dim_ff=3,
    # dropout=0.2,
    seq_len=10,
    # dropout=0.5,
    # dropout=0.4,
    # dropout=0.3,
    # dropout=0.15,
    # dropout=0.1,
    # dropout=0.05,
    dropout=0,
    # lr=1e-3,
    # lr=1e-5,
    # lr=7.5*1e-4,
    # lr=5*1e-4,
    # lr=2.5*1e-4,
    # lr=1.25*1e-4,
    )

convTran_3ff_5len: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    lr=1e-4,
    dim_ff=3,
    dropout=0.2,
    seq_len=5,
    )
# Object used in main.py file.
# convTranConfig: ConvTranConfig = convTranDefault

# convTranConfig: ConvTranConfig = convTran_100ff
# convTranConfig: ConvTranConfig = convTran_50ff
# convTranConfig: ConvTranConfig = convTran_25ff

# convTranConfig: ConvTranConfig = convTran_16ff
# convTranConfig: ConvTranConfig = convTran_10ff
# convTranConfig: ConvTranConfig = convTran_8ff

# convTranConfig: ConvTranConfig = convTran_6ff
# convTranConfig: ConvTranConfig = convTran_4ff
# convTranConfig: ConvTranConfig = convTran_3ff

# convTranConfig: ConvTranConfig = convTran_2ff
# convTranConfig: ConvTranConfig = convTran_1ff

# convTranConfig = convTran_3ff_3channel
# convTranConfig = convTran_3ff_2channel
# convTranConfig = convTran_3ff_1channel

# convTranConfig = convTran_2ff_3channel
# convTranConfig = convTran_2ff_2channel
# convTranConfig = convTran_2ff_1channel

# convTranConfig = convTran_2ff_2channel_8emb
# convTranConfig = convTran_2ff_2channel_6emb
# convTranConfig = convTran_3ff_2channel_8emb

# convTranConfig = convTran_2ff_2channel_4head
# convTranConfig = convTran_2ff_2channel_3head
# convTranConfig = convTran_2ff_2channel_2head
# convTranConfig = convTran_2ff_2channel_1head

# convTranConfig = convTran_2ff_2channel_8emb_120len
# convTranConfig = convTran_2ff_2channel_8emb_60len
# convTranConfig = convTran_2ff_2channel_8emb_50len

# convTranConfig = convTran_2ff_2channel_8emb_40len
convTranConfig = convTran_2ff_2channel_8emb_30len
# convTranConfig = convTran_2ff_2channel_8emb_25len

# convTranConfig = convTran_2ff_2channel_8emb_20len
# convTranConfig = convTran_2ff_2channel_8emb_15len
# convTranConfig = convTran_2ff_2channel_8emb_10len

# convTranConfig = convTran_2ff_2channel_8emb_5len

# convTranConfig: ConvTranConfig = convTran_3ff_120len
# convTranConfig: ConvTranConfig = convTran_3ff_60len
# convTranConfig: ConvTranConfig = convTran_3ff_50len
# convTranConfig: ConvTranConfig = convTran_3ff_40len
# convTranConfig: ConvTranConfig = convTran_3ff_30len
# convTranConfig: ConvTranConfig = convTran_3ff_20len
# convTranConfig: ConvTranConfig = convTran_3ff_15len
# convTranConfig: ConvTranConfig = convTran_3ff_10len
# convTranConfig: ConvTranConfig = convTran_3ff_5len

# convTranConfig: ConvTranConfig = convTran_3ff_10len


################################################################################
# iTransformer
################################################################################
@dataclass
class ITransformerConfig(BaseConfig):
    """Configuration class for the iTransformer model. Default hyperparameters
    are as given in the official implementation, which assumes a sequence length
    (seq_len) of 1000.
    https://github.com/thuml/iTransformer/blob/main/run.py

    Important variable name changes
    -------------------------------
    emb_size:
        Originally called d_model in the offical implementation and paper. 
        Changed to emb_size to match the same naming conventaion as other models
        and to avoid confusion.
    """
    # optional model spec
    dim_ff: int = 2048 # dimension of dense feedforward part of transformer layer
    emb_size: int = 512 # internal dimension of transformer embeddings
    num_heads: int = 8 # number of multi-headed attention heads
    e_layers: int = 2  # number of encoder layers
    dropout: float = 0.1
    activation: str = 'gelu'
    use_norm: bool = False  # whether to normalise input values.

    # optimiser settings
    optimiser: str | Callable = 'RAdam'
    lr: float = 1e-4


# Objects
iTransformerDefault: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    # dropout=0.2,
    # dropout=0.3,
)


iTransformer_256ff: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 256,
)

iTransformer_200ff: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 200,
)


iTransformer_200ff_50emb: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 200,
    emb_size = 50,
    dropout=0.2,
)


iTransformer_100ff_50emb: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 100,
    emb_size = 50,
    # dropout=0.2,
)


iTransformer_50ff_50emb: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 50,
    emb_size = 50,
    # dropout=0.2,
)


iTransformer_100ff_25emb: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 100,
    emb_size = 25,
    # lr = 0.001,
    # dropout=0.3,
)


iTransformer_50ff_25emb: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 50,
    emb_size = 25,
    # lr = 0.001,
    # dropout=0.3
)


iTransformer_50ff_12emb: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 50,
    emb_size = 12,
    # lr = 0.001
)

# interested in below 3
iTransformer_25ff_12emb: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 25,
    emb_size = 12,
    # lr = 1e-3,
    # lr = 5*1e-4,
    # lr = 1e-4,
    # dropout = 0.15,
    # dropout = 0.125,
    # dropout = 0.1,
)


iTransformer_20ff_10emb: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 20,
    emb_size = 10,
    # lr = 1e-3,
    # lr = 5*1e-4,
    # lr = 1e-4,
    # dropout = 0.15,
    # dropout = 0.125,
    # dropout = 0.1,
)


iTransformer_10ff_10emb: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 10,
    emb_size = 10,
    # lr = 1e-3,
    # lr = 5*1e-4,
    # lr = 1e-4,
    # dropout = 0.15,
    # dropout = 0.125,
    # dropout = 0.1,
)

iTransformer_5ff_10emb: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 5,
    emb_size = 10,
    # lr = 1e-3,
    # lr = 5*1e-4,
    # lr = 1e-4,
    # dropout = 0.15,
    # dropout = 0.125,
    # dropout = 0.1,
)

iTransformer_4ff_10emb: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 4,
    emb_size = 10,
    # lr = 1e-3,
    # lr = 5*1e-4,
    # lr = 1e-4,
    # dropout = 0.15,
    # dropout = 0.125,
    # dropout = 0.1,
)

iTransformer_3ff_10emb: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size = 10,
    # lr = 1e-3,
    # lr = 5*1e-4,
    # lr = 1e-4,
    # dropout = 0.15,
    # dropout = 0.125,
    # dropout = 0.1,
)

iTransformer_2ff_10emb: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 2,
    emb_size = 10,
    # lr = 1e-3,
    # lr = 5*1e-4,
    # lr = 1e-4,
    # dropout = 0.15,
    # dropout = 0.125,
    # dropout = 0.1,
)

iTransformer_1ff_10emb: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 1,
    emb_size = 10,
    # lr = 1e-3,
    # lr = 5*1e-4,
    # lr = 1e-4,
    # dropout = 0.15,
    # dropout = 0.125,
    # dropout = 0.1,
)

iTransformer_3ff_5emb: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 2,
    emb_size = 5,
    # lr = 1e-3,
    # lr = 5*1e-4,
    # lr = 1e-4,
    # dropout = 0.15,
    # dropout = 0.125,
    # dropout = 0.1,
)

# seq len study
iTransformer_3ff_10emb_120len: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size = 10,
    seq_len=120,
)

iTransformer_3ff_10emb_60len: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size = 10,
    seq_len=60,
)

iTransformer_3ff_10emb_50len: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size = 10,
    seq_len=50,
)

iTransformer_3ff_10emb_40len: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size = 10,
    seq_len=40,
)

iTransformer_3ff_10emb_30len: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    # lr = 1e-2,
    # lr = 1e-3,
    # lr = 1e-5,
    # lr = 7.5*1e-4,
    # lr = 5*1e-4,
    lr = 2.5*1e-4,
    # dropout=0.5,
    # dropout=0.4,
    # dropout=0.3,
    dropout=0.2,
    # dropout=0.175,
    # dropout=0.15,
    # dropout=0.125,
    # dropout=0.05,
    # dropout=0,
)

iTransformer_3ff_10emb_20len: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size = 10,
    seq_len=20,
)

iTransformer_3ff_10emb_15len: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size = 10,
    seq_len=15,
)

iTransformer_3ff_10emb_10len: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size = 10,
    seq_len=10,
)

iTransformer_3ff_10emb_5len: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size = 10,
    seq_len=5,
)

iTransformer_3ff_10emb_30len_2heads: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    # lr = 1e-2,
    # lr = 1e-3,
    # lr = 1e-5,
    # lr = 7.5*1e-4,
    # lr = 5*1e-4,
    # lr = 2.5*1e-4,
    # dropout=0.5,
    # dropout=0.4,
    # dropout=0.3,
    dropout=0.2,
    # dropout=0.175,
    # dropout=0.15,
    # dropout=0.125,
    # dropout=0.05,
    # dropout=0,
    num_heads=2,
)

iTransformer_3ff_10emb_30len_2heads_1e: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    # lr = 1e-2,
    # lr = 1e-3,
    # lr = 1e-5,
    # lr = 7.5*1e-4,
    # lr = 5*1e-4,
    # lr = 2.5*1e-4,
    # dropout=0.5,
    # dropout=0.4,
    # dropout=0.3,
    dropout=0.2,
    # dropout=0.175,
    # dropout=0.15,
    # dropout=0.125,
    # dropout=0.05,
    # dropout=0,
    num_heads=2,
    e_layers=1,
)

iTransformer_3ff_10emb_30len_1heads_1e: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    # lr = 1e-2,
    # lr = 1e-3,
    # lr = 7.5*1e-4,
    # lr = 5*1e-4,
    lr = 2.5*1e-4,
    # lr = 7.5*1e-5,
    # lr = 5*1e-5,
    # dropout=0.5,
    # dropout=0.4,
    # dropout=0.3,
    # dropout=0.25,
    # dropout=0.2,
    # dropout=0.175,
    # dropout=0.15,
    # dropout = 0.1,
    # dropout=0.125,
    # dropout=0.05,
    dropout=0,
    num_heads=1,
    e_layers=1,
)

# Object used in main.py file.
# iTransformerConfig: ITransformerConfig = iTransformerDefault
# iTransformerConfig: ITransformerConfig = iTransformer_256ff
# iTransformerConfig: ITransformerConfig = iTransformer_200ff
# iTransformerConfig: ITransformerConfig = iTransformer_200ff_50emb

# iTransformerConfig: ITransformerConfig = iTransformer_100ff_50emb
# iTransformerConfig: ITransformerConfig = iTransformer_50ff_50emb
# iTransformerConfig: ITransformerConfig = iTransformer_100ff_25emb
# iTransformerConfig: ITransformerConfig = iTransformer_50ff_25emb

# iTransformerConfig: ITransformerConfig = iTransformer_50ff_12emb
# iTransformerConfig: ITransformerConfig = iTransformer_25ff_12emb
# iTransformerConfig: ITransformerConfig = iTransformer_20ff_10emb
# iTransformerConfig: ITransformerConfig = iTransformer_10ff_10emb

# iTransformerConfig: ITransformerConfig = iTransformer_5ff_10emb

# iTransformerConfig: ITransformerConfig = iTransformer_3ff_10emb
# iTransformerConfig: ITransformerConfig = iTransformer_2ff_10emb
# iTransformerConfig: ITransformerConfig = iTransformer_1ff_10emb
# iTransformerConfig: ITransformerConfig = iTransformer_4ff_10emb

# iTransformerConfig: ITransformerConfig = iTransformer_3ff_10emb_120len
# iTransformerConfig: ITransformerConfig = iTransformer_3ff_10emb_60len
# iTransformerConfig: ITransformerConfig = iTransformer_3ff_10emb_50len
# iTransformerConfig: ITransformerConfig = iTransformer_3ff_10emb_40len

# iTransformerConfig: ITransformerConfig = iTransformer_3ff_10emb_30len
# iTransformerConfig: ITransformerConfig = iTransformer_3ff_10emb_20len
# iTransformerConfig: ITransformerConfig = iTransformer_3ff_10emb_10len
# iTransformerConfig: ITransformerConfig = iTransformer_3ff_10emb_5len

# iTransformerConfig: ITransformerConfig = iTransformer_3ff_10emb_15len

# iTransformerConfig: ITransformerConfig = iTransformer_3ff_10emb_30len
# iTransformerConfig = iTransformer_3ff_10emb_30len_2heads
# iTransformerConfig = iTransformer_3ff_10emb_30len_2heads_1e
iTransformerConfig = iTransformer_3ff_10emb_30len_1heads_1e
# ################################################################################
# Bi-Mamba
################################################################################
@dataclass
class BiMambaConfig(BaseConfig):
    """Configuration class for the Bi-Mamba model.
    """
    encoder_name: str = _no_default
    block_name: str = 'mamba'
    e_layers: int = 2 # changed from n_layers
    dropout: float = 0.2
    patching: str | None = None

    # Optimiser settings
    optimiser: str | Callable = torch.optim.Adam
    lr: float = 0.001

    # Encoder block settings
    dt_rank: int | str = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4
    rms_norm_eps: float = 1e-5
    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False # apply layernorms to internal activations
    pscan: bool = True # use parallel scan mode or sequential mode when training
    use_cuda: bool = False # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        super(). __post_init__()
        self.encoder_name = self.encoder_name.lower()

        if self.emb_size is _no_default:
            self.emb_size = self.d_model

        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

# Objects
biMambaUnivariateDailyFXConfig: BiMambaConfig = BiMambaConfig(
    dataset_name='univariate_fx_daily',
    encoder_name='BiMamba+',
    optimiser=torch.optim.Adam,
)

biMambaFusionUnivariateDailyFXConfig: BiMambaConfig = BiMambaConfig(
    dataset_name='univariate_fx_daily',
    encoder_name='BiMambaFusion',
    optimiser=torch.optim.Adam
)

# Object used in main.py file.
biMambaConfig: BiMambaConfig = biMambaUnivariateDailyFXConfig
biMambaFusionConfig: BiMambaConfig = biMambaFusionUnivariateDailyFXConfig

################################################################################
# ConvMamba
################################################################################
@dataclass
class ConvMambaConfig(BaseConfig):
    """Configuration class for the ConvMamba model.
    """
    block_name: str = 'mamba'
    e_layers: int = 2
    dropout: float = 0.2
    emb_size: int = 16
    dim_ff: int = 256
    channel_expand: int = 4
    optimiser: str | Callable = 'RAdam'
    residuals: bool = True

    # Optimiser settings
    lr: float = 1e-4

    # Encoder block settings
    dt_rank: int | str = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4
    rms_norm_eps: float = 1e-5
    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False # apply layernorms to internal activations
    pscan: bool = True # use parallel scan mode or sequential mode when training
    use_cuda: bool = False # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        super(). __post_init__()

        # This annoying rename is necessary in order to make our config work with the mambapy mamba block
        self.nvars = self.d_model
        self.d_model = self.emb_size

        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

# Objects
convMambaDefault: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',

)


convMamba_100ff: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 100,
)

convMamba_50ff: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 50,
)

convMamba_25ff: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 25,
)

convMamba_25ff_10emb: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 25,
    emb_size=10,
)

convMamba_25ff_10emb_10state: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 25,
    emb_size=10,
    d_state=10,
)

convMamba_25ff_10emb_10state_1e: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 25,
    emb_size=10,
    d_state=10,
    e_layers=1,
)


convMamba_10ff_5emb_5state: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 10,
    emb_size=5,
    d_state=5,
)

convMamba_8ff_4emb_4state: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 8,
    emb_size=4,
    d_state=4,
    # dropout=0.1,
    # dropout=0.15,
    # dropout=0.125,
    # e_layers=1,
    # lr=1e-4,
    # lr=1.1e-4,
    # lr=9e-5,
    # lr=8e-5,
    # lr=7e-5,
    # lr=6e-5,
    # lr=5e-5,
)


convMamba_8ff_4emb_4state_1e: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 8,
    emb_size=4,
    d_state=4,
    e_layers=1,
    # lr=1e-4,
    # dropout=0.125,
    # lr=1e-4,
    # lr=5*1e-5,
    # lr=7.5*1e-5,
    # lr=8*1e-5,
    # lr=9*1e-5,
)

convMamba_6ff_3emb_3state: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 6,
    emb_size=3,
    d_state=3,
    # # e_layers=1,
    # lr=1e-4,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_4ff_4emb_4state: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 4,
    emb_size=4,
    d_state=4,
    # # e_layers=1,
    # lr=1e-4,
    # # dropout=0.1,
    # # dropout=0.15,
    # dropout=0.125,
)

convMamba_6ff_3emb_3state_1e: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 6,
    emb_size=3,
    d_state=3,
    e_layers=1,
    lr=1e-4,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_4ff_4emb_4state_1e: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 4,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_4ff_3emb_3state_1e: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 4,
    emb_size=3,
    d_state=3,
    e_layers=1,
    lr=1e-4,
    # # dropout=0.1,
    # dropout=0.15,
)


convMamba_3ff_3emb_3state_1e: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=3,
    d_state=3,
    e_layers=1,
    lr=1e-4,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_4emb_4state_1e: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_3emb_3state_1e_3channel: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=3,
    d_state=3,
    e_layers=1,
    lr=1e-4,
    channel_expand=3,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_4emb_4state_1e_3channel: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=3,
    # # dropout=0.1,
    # dropout=0.15,
)


convMamba_3ff_3emb_3state_1e_2channel: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=3,
    d_state=3,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_4emb_4state_1e_2channel: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_4emb_4state_1e_1channel: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=1,
    # # dropout=0.1,
    # dropout=0.15,
)
# seq len study
convMamba_6ff_3emb_3state_120len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 6,
    emb_size=3,
    d_state=3,
    seq_len=120,
)

convMamba_6ff_3emb_3state_60len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 6,
    emb_size=3,
    d_state=3,
    seq_len=60,
)

convMamba_6ff_3emb_3state_50len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 6,
    emb_size=3,
    d_state=3,
    seq_len=50,
)

convMamba_6ff_3emb_3state_40len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 6,
    emb_size=3,
    d_state=3,
    seq_len=40,
)

convMamba_6ff_3emb_3state_30len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 6,
    emb_size=3,
    d_state=3,
    seq_len=30,
)

convMamba_6ff_3emb_3state_20len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 6,
    emb_size=3,
    d_state=3,
    seq_len=20,
)

convMamba_6ff_3emb_3state_15len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 6,
    emb_size=3,
    d_state=3,
    seq_len=15,
)

convMamba_6ff_3emb_3state_10len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 6,
    emb_size=3,
    d_state=3,
    seq_len=10,
)

convMamba_6ff_3emb_3state_5len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 6,
    emb_size=3,
    d_state=3,
    seq_len=5,
)

##
convMamba_10ff_5emb_5state_120len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 10,
    emb_size=5,
    d_state=5,
    seq_len=120,
)

convMamba_10ff_5emb_5state_60len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 10,
    emb_size=5,
    d_state=5,
    seq_len=60,
)

convMamba_10ff_5emb_5state_50len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 10,
    emb_size=5,
    d_state=5,
    seq_len=50,
)

convMamba_10ff_5emb_5state_40len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 10,
    emb_size=5,
    d_state=5,
    seq_len=40,
)

convMamba_10ff_5emb_5state_30len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 10,
    emb_size=5,
    d_state=5,
    seq_len=30,
)

convMamba_10ff_5emb_5state_20len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 10,
    emb_size=5,
    d_state=5,
    seq_len=20,
)

convMamba_10ff_5emb_5state_15len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 10,
    emb_size=5,
    d_state=5,
    seq_len=15,
)

convMamba_10ff_5emb_5state_10len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 10,
    emb_size=5,
    d_state=5,
    seq_len=10,
)

convMamba_10ff_5emb_5state_5len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 10,
    emb_size=5,
    d_state=5,
    seq_len=5,
)

convMamba_10ff_5emb_5state_1e_30len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 10,
    emb_size=5,
    d_state=5,
    seq_len=30,
    e_layers=1,
)
###

convMamba_8ff_4emb_4state_1e_60len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 8,
    emb_size=4,
    d_state=4,
    seq_len=60,
    e_layers=1,
)

convMamba_8ff_4emb_4state_1e_50len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 8,
    emb_size=4,
    d_state=4,
    seq_len=50,
    e_layers=1,
)

convMamba_8ff_4emb_4state_1e_40len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 8,
    emb_size=4,
    d_state=4,
    seq_len=40,
    e_layers=1,
)

convMamba_8ff_4emb_4state_1e_30len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 8,
    emb_size=4,
    d_state=4,
    seq_len=30,
    e_layers=1,
    # lr=1e-3,
    # lr=5*1e-4,
    # lr=5*1e-5,
    # lr=1e-5,
    lr=2.5*1e-4,
    # dropout=0.5,
    # dropout=0.4,
    # dropout=0.3,
    # dropout=0.25,
    # dropout=0.15,
    # dropout=0.1,
    # dropout=0.05,
    dropout=0,
)

convMamba_8ff_4emb_4state_1e_25len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 8,
    emb_size=4,
    d_state=4,
    seq_len=25,
    e_layers=1,
)

convMamba_8ff_4emb_4state_1e_20len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 8,
    emb_size=4,
    d_state=4,
    seq_len=20,
    e_layers=1,
)

convMamba_8ff_4emb_4state_1e_15len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 8,
    emb_size=4,
    d_state=4,
    seq_len=15,
    e_layers=1,
)

convMamba_8ff_4emb_4state_1e_10len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 8,
    emb_size=4,
    d_state=4,
    seq_len=10,
    e_layers=1,
)

convMamba_8ff_4emb_4state_1e_5len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 8,
    emb_size=4,
    d_state=4,
    seq_len=5,
    e_layers=1,
)

convMamba_6ff_3emb_3state_30len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 6,
    emb_size=3,
    d_state=3,
    seq_len=30,
    lr=2.5*1e-4,
)

convMamba_6ff_3emb_3state_1e_30len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 6,
    emb_size=3,
    d_state=3,
    seq_len=30,
    e_layers=1,
    lr=2.5*1e-4,
)

######## new seq len
convMamba_3ff_3emb_3state_1e_120len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=3,
    d_state=3,
    e_layers=1,
    lr=1e-4,
    seq_len=120,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_3emb_3state_1e_60len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=3,
    d_state=3,
    e_layers=1,
    lr=1e-4,
    seq_len=60,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_3emb_3state_1e_50len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=3,
    d_state=3,
    e_layers=1,
    lr=1e-4,
    seq_len=50,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_3emb_3state_1e_40len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=3,
    d_state=3,
    e_layers=1,
    lr=1e-4,
    seq_len=40,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_3emb_3state_1e_30len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=3,
    d_state=3,
    e_layers=1,
    lr=1e-4,
    seq_len=30,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_3emb_3state_1e_25len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=3,
    d_state=3,
    e_layers=1,
    lr=1e-4,
    seq_len=25,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_3emb_3state_1e_20len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=3,
    d_state=3,
    e_layers=1,
    lr=1e-4,
    seq_len=20,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_3emb_3state_1e_15len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=3,
    d_state=3,
    e_layers=1,
    lr=1e-4,
    seq_len=15,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_3emb_3state_1e_10len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=3,
    d_state=3,
    e_layers=1,
    lr=1e-4,
    seq_len=10,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_3emb_3state_1e_5len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=3,
    d_state=3,
    e_layers=1,
    lr=1e-4,
    seq_len=5,
    # # dropout=0.1,
    # dropout=0.15,
)

########
convMamba_3ff_4emb_4state_1e_2channel_120len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=120,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_4emb_4state_1e_2channel_60len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=60,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_4emb_4state_1e_2channel_50len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=50,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_4emb_4state_1e_2channel_40len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=40,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_4emb_4state_1e_2channel_30len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    # lr=1e-3,
    # lr=7.5*1e-4,
    # lr=5*1e-4,
    # lr=2.5*1e-4,
    # lr=7.5*1e-5,
    # lr=5*1e-5,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
    # dropout=0.5,
    # dropout=0.4,
    # dropout=0.3,
    # dropout=0.25,
    # dropout=0.15,
    dropout=0.1,
    # dropout=0.05,
    # dropout=0,
)

convMamba_3ff_4emb_4state_1e_2channel_25len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=25,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_4emb_4state_1e_2channel_20len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=20,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_4emb_4state_1e_2channel_15len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=15,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_4emb_4state_1e_2channel_10len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=10,
    # # dropout=0.1,
    # dropout=0.15,
)

convMamba_3ff_4emb_4state_1e_2channel_5len: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=5,
    # # dropout=0.1,
    # dropout=0.15,
)


# convMambaConfig: ConvMambaConfig = convMamba_6ff_3emb_3state_30len
# convMambaConfig: ConvMambaConfig = convMamba_6ff_3emb_3state_1e_30len

# Object used in main.py file.
# convMambaConfig: ConvMambaConfig = convMambaDefault
# convMambaConfig: ConvMambaConfig = convMamba_100ff
# convMambaConfig: ConvMambaConfig = convMamba_50ff

# convMambaConfig: ConvMambaConfig = convMamba_25ff
# convMambaConfig: ConvMambaConfig = convMamba_25ff_10emb
# convMambaConfig: ConvMambaConfig = convMamba_25ff_10emb_10state
# convMambaConfig: ConvMambaConfig = convMamba_25ff_10emb_10state_1e

# convMambaConfig: ConvMambaConfig = convMamba_10ff_5emb_5state
# convMambaConfig: ConvMambaConfig = convMamba_8ff_4emb_4state
# convMambaConfig: ConvMambaConfig = convMamba_8ff_4emb_4state_1e
# convMambaConfig: ConvMambaConfig = convMamba_6ff_3emb_3state

# convMambaConfig: ConvMambaConfig = convMamba_4ff_4emb_4state

# convMambaConfig: ConvMambaConfig = convMamba_6ff_3emb_3state_120len
# convMambaConfig: ConvMambaConfig = convMamba_6ff_3emb_3state_60len
# convMambaConfig: ConvMambaConfig = convMamba_6ff_3emb_3state_50len
# convMambaConfig: ConvMambaConfig = convMamba_6ff_3emb_3state_40len

# convMambaConfig: ConvMambaConfig = convMamba_6ff_3emb_3state_30len
# convMambaConfig: ConvMambaConfig = convMamba_6ff_3emb_3state_20len
# convMambaConfig: ConvMambaConfig = convMamba_6ff_3emb_3state_15len
# convMambaConfig: ConvMambaConfig = convMamba_6ff_3emb_3state_10len
# convMambaConfig: ConvMambaConfig = convMamba_6ff_3emb_3state_5len

# convMambaConfig: ConvMambaConfig = convMamba_10ff_5emb_5state_60len
# convMambaConfig: ConvMambaConfig = convMamba_10ff_5emb_5state_50len
# convMambaConfig: ConvMambaConfig = convMamba_10ff_5emb_5state_40len
# convMambaConfig: ConvMambaConfig = convMamba_10ff_5emb_5state_30len
# convMambaConfig: ConvMambaConfig = convMamba_10ff_5emb_5state_20len
# convMambaConfig: ConvMambaConfig = convMamba_10ff_5emb_5state_15len
# convMambaConfig: ConvMambaConfig = convMamba_10ff_5emb_5state_10len
# convMambaConfig: ConvMambaConfig = convMamba_10ff_5emb_5state_5len

# convMambaConfig: ConvMambaConfig = convMamba_10ff_5emb_5state_1e_30len

# convMambaConfig: ConvMambaConfig = convMamba_8ff_4emb_4state_1e_30len

# convMambaConfig: ConvMambaConfig = convMamba_8ff_4emb_4state_1e_60len
# convMambaConfig: ConvMambaConfig = convMamba_8ff_4emb_4state_1e_50len
# convMambaConfig: ConvMambaConfig = convMamba_8ff_4emb_4state_1e_40len
# convMambaConfig: ConvMambaConfig = convMamba_8ff_4emb_4state_1e_20len

# convMambaConfig: ConvMambaConfig = convMamba_8ff_4emb_4state_1e_25len
# convMambaConfig: ConvMambaConfig = convMamba_8ff_4emb_4state_1e_15len
# convMambaConfig: ConvMambaConfig = convMamba_8ff_4emb_4state_1e_10len
# convMambaConfig: ConvMambaConfig = convMamba_8ff_4emb_4state_1e_5len

# convMambaConfig: ConvMambaConfig = convMamba_6ff_3emb_3state_1e
# convMambaConfig: ConvMambaConfig = convMamba_4ff_4emb_4state_1e
# convMambaConfig: ConvMambaConfig = convMamba_4ff_3emb_3state_1e
# convMambaConfig: ConvMambaConfig = convMamba_3ff_3emb_3state_1e
# convMambaConfig: ConvMambaConfig = convMamba_3ff_4emb_4state_1e

# convMambaConfig: ConvMambaConfig = convMamba_3ff_3emb_3state_1e_3channel
# convMambaConfig: ConvMambaConfig = convMamba_3ff_4emb_4state_1e_3channel

# convMambaConfig: ConvMambaConfig = convMamba_3ff_3emb_3state_1e_2channel
# convMambaConfig: ConvMambaConfig = convMamba_3ff_4emb_4state_1e_2channel
# convMambaConfig: ConvMambaConfig = convMamba_3ff_4emb_4state_1e_1channel

# convMambaConfig: ConvMambaConfig = convMamba_3ff_3emb_3state_1e_120len
# convMambaConfig: ConvMambaConfig = convMamba_3ff_3emb_3state_1e_60len
# convMambaConfig: ConvMambaConfig = convMamba_3ff_3emb_3state_1e_50len

# convMambaConfig: ConvMambaConfig = convMamba_3ff_3emb_3state_1e_40len
# convMambaConfig: ConvMambaConfig = convMamba_3ff_3emb_3state_1e_30len
# convMambaConfig: ConvMambaConfig = convMamba_3ff_3emb_3state_1e_25len

# convMambaConfig: ConvMambaConfig = convMamba_3ff_3emb_3state_1e_20len
# convMambaConfig: ConvMambaConfig = convMamba_3ff_3emb_3state_1e_15len
# convMambaConfig: ConvMambaConfig = convMamba_3ff_3emb_3state_1e_10len
# convMambaConfig: ConvMambaConfig = convMamba_3ff_3emb_3state_1e_5len

# convMambaConfig: ConvMambaConfig = convMamba_3ff_4emb_4state_1e_2channel_120len
# convMambaConfig: ConvMambaConfig = convMamba_3ff_4emb_4state_1e_2channel_60len
# convMambaConfig: ConvMambaConfig = convMamba_3ff_4emb_4state_1e_2channel_50len
# convMambaConfig: ConvMambaConfig = convMamba_3ff_4emb_4state_1e_2channel_40len
convMambaConfig: ConvMambaConfig = convMamba_3ff_4emb_4state_1e_2channel_30len
# convMambaConfig: ConvMambaConfig = convMamba_3ff_4emb_4state_1e_2channel_25len

# convMambaConfig: ConvMambaConfig = convMamba_3ff_4emb_4state_1e_2channel_20len
# convMambaConfig: ConvMambaConfig = convMamba_3ff_4emb_4state_1e_2channel_15len
# convMambaConfig: ConvMambaConfig = convMamba_3ff_4emb_4state_1e_2channel_10len
# convMambaConfig: ConvMambaConfig = convMamba_3ff_4emb_4state_1e_2channel_5len
################################################################################
# Mamba
################################################################################
@dataclass
class MambaConfig(BaseConfig):
    block_name: str = 'mamba'
    e_layers: int = 2 # changed from n_layers
    dropout: float = 0.2
    dim_ff: int = 256
    residuals: bool = False

    # Optimiser settings
    lr: float = 0.001
    optimiser: str | Callable = 'RAdam'

    # Encoder block settings
    dt_rank: int | str = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4
    dt_min: float = 1e-3
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4
    rms_norm_eps: float = 1e-5
    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False # apply layernorms to internal activations
    pscan: bool = True # use parallel scan mode or sequential mode when training
    use_cuda: bool = False # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        super(). __post_init__()

        # This annoying rename is necessary in order to make our config work with the mambapy mamba block
        self.emb_size = self.d_model

        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

# Objects
mambaDefault: MambaConfig = MambaConfig(
    dataset_name='univariate_fx_daily',
    # lr=1e-2,
    # lr=1e-4,
    # lr=1e-5,
    residuals=True,
    # dropout=0.5,
    # dropout=0.4,
    # dropout=0.3,
    # dropout=0.1,
    dropout=0,
)

mamba_ff = MambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=1000,
    residuals=True,
)

mamba_state = MambaConfig(
    dataset_name='univariate_fx_daily',
    d_state = 64,
    residuals=True,
)

mamba_encoder = MambaConfig(
    dataset_name='univariate_fx_daily',
    e_layers=8,
    residuals=True,
)

mamba_len = MambaConfig(
    dataset_name='univariate_fx_daily',
    seq_len=5,
    residuals=True,
)

# mamba_2kff: MambaConfig = MambaConfig(
#     dataset_name='univariate_fx_daily',
#     # lr=1e-4,
#     dim_ff=2000,
#     residuals=True,
# )

# mamba_8e: MambaConfig = MambaConfig(
#     dataset_name='univariate_fx_daily',
#     # lr=1e-4,
#     residuals=True,
# )

# mamba_500ff_8e: MambaConfig = MambaConfig(
#     dataset_name='univariate_fx_daily',
#     dim_ff=500,
#     # lr=1e-4,
#     residuals=True,
#     e_layers=8,
# )

# mamba_500ff_8e_32state: MambaConfig = MambaConfig(
#     dataset_name='univariate_fx_daily',
#     dim_ff=500,
#     # lr=1e-4,
#     residuals=True,
#     e_layers=8,
#     d_state=32,
# )

# mamba_500ff_8e_64state: MambaConfig = MambaConfig(
#     dataset_name='univariate_fx_daily',
#     dim_ff=500,
#     # lr=1e-4,
#     residuals=True,
#     e_layers=8,
#     d_state=64,
# )

# mambaDefault_30len: MambaConfig = MambaConfig(
#     dataset_name='univariate_fx_daily',
#     lr=1e-3,
#     residuals=True,
#     # dropout=0.5,
#     # dropout=0.4,
#     # dropout=0.3,
#     # dropout=0.1,
#     # dropout=0,
#     seq_len=30,
# )

# Object used in main.py file.
mambaConfig: MambaConfig = mambaDefault
# mambaConfig: MambaConfig = mamba_ff
# mambaConfig: MambaConfig = mamba_state
# mambaConfig: MambaConfig = mamba_encoder
# mambaConfig: MambaConfig = mamba_len

# mambaConfig: MambaConfig = mambaDefault_30len
# mambaConfig: MambaConfig = mamba_2kff
# mambaConfig: MambaConfig = mamba_8e

# mambaConfig: MambaConfig = mamba_500ff_8e
# mambaConfig: MambaConfig = mamba_500ff_8e_32state
# mambaConfig: MambaConfig = mamba_500ff_8e_64state

