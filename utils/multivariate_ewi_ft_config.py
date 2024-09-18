from utils.config import ConvMambaConfig, ITransformerConfig, ConvTranConfig, MambaConfig

dataset = 'multivariate_fx_daily'

################################################################################
# iTransformer
################################################################################
iTransformerDefault: ITransformerConfig = ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,

    # lr = 1e-3,
    # lr=7.5*1e-4,
    # lr = 5*1e-4,

    # lr=1e-4,
    # lr=7.5*1e-5,
    # lr=5*1e-5,


    # dropout=0.05,
    # dropout=0.1,
    # dropout=0.15,
    
    # dropout = 0.25,
    # dropout=0.3,
    # dropout=0.35,

    num_heads=1,
    e_layers=1,
)

iTransformer_15emb= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 15,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=1,
    e_layers=1,
)

iTransformer_20emb= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 20,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=1,
    e_layers=1,
)

iTransformer_25emb= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 25,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=1,
    e_layers=1,
)

iTransformer_2e= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=1,
    e_layers=2,
)

iTransformer_3e= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=1,
    e_layers=3,
)

iTransformer_4e= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=1,
    e_layers=4,
)

iTransformer_2heads= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=2,
    e_layers=1,
)

iTransformer_3heads= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=3,
    e_layers=1,
)

iTransformer_4heads= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=4,
    e_layers=1,
)

iTransformer_4ff= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 4,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=1,
    e_layers=1,
)

iTransformer_5ff= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 5,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=1,
    e_layers=1,
)

iTransformer_6ff= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 6,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=1,
    e_layers=1,
)
iTransformerConfig = iTransformerDefault
# iTransformerConfig = iTransformer_15emb
# iTransformerConfig = iTransformer_20emb
# iTransformerConfig = iTransformer_25emb

# iTransformerConfig = iTransformer_2e
# iTransformerConfig = iTransformer_3e
# iTransformerConfig = iTransformer_4e

# iTransformerConfig = iTransformer_2heads
# iTransformerConfig = iTransformer_3heads
# iTransformerConfig = iTransformer_4heads

# iTransformerConfig = iTransformer_4ff
# iTransformerConfig = iTransformer_5ff
# iTransformerConfig = iTransformer_6ff

# seq len
iTransformer_35len= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=35,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=1,
    e_layers=1,
)

iTransformer_40len= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=40,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=1,
    e_layers=1,
)

iTransformer_45len= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=45,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=1,
    e_layers=1,
)

iTransformer_25len= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=25,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=1,
    e_layers=1,
)

iTransformer_20len= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=20,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=1,
    e_layers=1,
)

iTransformer_15len= ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=15,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=1,
    e_layers=1,
)

# iTransformerConfig = iTransformer_35len
# iTransformerConfig = iTransformer_40len
# iTransformerConfig = iTransformer_45len
# iTransformerConfig = iTransformer_15len
# iTransformerConfig = iTransformer_20len
# iTransformerConfig = iTransformer_25len
################################################################################
# ConvTran
################################################################################
convTranDefault: ConvTranConfig = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=30,
    # lr=1e-3,
    # dropout=0.2,
    # dropout=0.3,
    # dropout=0.4,
    # dropout=0.5,

    # dropout=0.6,
    # dropout=0.7,
    # dropout=0.8,
    # dropout=0.9,

    dropout=0.95,

    # lr=7.5*1e-4,
    # lr=5*1e-4,
    # lr=2.5*1e-4,
    # lr=1e-4,

    lr=7.5*1e-5,
    # lr=5*1e-5,
    # lr=2.5*1e-5,
    # lr=1e-5,
    )

convTran_1channel = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=1,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
    # dropout=0.3,
    # dropout=0.4,
    # dropout=0.5,

    # dropout=0.6,
    # dropout=0.7,
    # dropout=0.8,
    # dropout=0.9,

    # dropout=0.95,

    # lr=7.5*1e-4,
    # lr=5*1e-4,
    # lr=2.5*1e-4,
    # lr=1e-4,

    # lr=7.5*1e-5,
    # lr=5*1e-5,
    # lr=2.5*1e-5,
    # lr=1e-5,
    )

convTran_1_4kernels = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 4),
)

convTran_1channel_1_4kernels = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=1,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 4),
)

convTran_1_2kernels = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 2),
)

convTran_1channel_1_2kernels = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=1,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 2),
)

convTran_1_16kernels = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 16),
)

convTran_1channel_1_16kernels = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=1,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 16),
)

convTran_1_30kernels = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 30),
)

convTran_1channel_1_30kernels = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=1,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 30),
)

convTran_1_12kernels = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 12),
)

convTran_1channel_1_12kernels = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=1,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 12),
)

convTran_1_18kernels = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 18),
)

convTran_1channel_1_18kernels = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=1,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 18),
)

convTran_1_10kernels = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 10),
)

convTran_1channel_1_10kernels = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=1,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 10),
)

convTran_25len = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=25,
    lr=1e-3,
    dropout=0.2,
)

convTran_20len = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=20,
    lr=1e-3,
    dropout=0.2,
)

convTran_15len = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=15,
    lr=1e-3,
    dropout=0.2,
)

convTran_10len = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=10,
    lr=1e-3,
    dropout=0.2,
)

convTran_5len = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=5,
    # lr=1e-3,
    dropout=0.2,

    # dropout=0.25,
    # dropout=0.3,
    # dropout=0.4,
    # dropout=0.5,

    # dropout=0.6,
    # dropout=0.7,
    # dropout=0.8,
    # dropout=0.9,

    # dropout=0.95,

    # lr=7.5*1e-4,
    # lr=5*1e-4,
    # lr=2.5*1e-4,
    # lr=1e-4,

    # lr=7.5*1e-5,
    # lr=5*1e-5,
    # lr=2.5*1e-5,
    lr=1e-5,
)

convTran_1_12kernels_25len = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=25,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 12),
)

convTran_1_12kernels_20len = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=20,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 12),
)

convTran_1_12kernels_15len = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=15,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 12),
)

convTran_1_12kernels_10len = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=10,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 12),
)

convTran_1_12kernels_5len = ConvTranConfig(
    dataset_name=dataset,
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=5,
    lr=1e-3,
    dropout=0.2,
    kernel_sizes = (1, 12),
)
convTranConfig = convTranDefault
# convTranConfig = convTran_1channel
# convTranConfig = convTran_1_4kernels
# convTranConfig = convTran_1channel_1_4kernels
# convTranConfig = convTran_1_2kernels
# convTranConfig = convTran_1channel_1_2kernels

# convTranConfig = convTran_1_16kernels
# convTranConfig = convTran_1channel_1_16kernels

# convTranConfig = convTran_1_30kernels
# convTranConfig = convTran_1channel_1_30kernels

# convTranConfig = convTran_1_18kernels
# convTranConfig = convTran_1channel_1_18kernels
# convTranConfig = convTran_1_12kernels

# convTranConfig = convTran_1channel_1_12kernels

# convTranConfig = convTran_1_10kernels
# convTranConfig = convTran_1channel_1_10kernels

# convTranConfig = convTran_25len
# convTranConfig = convTran_20len
# convTranConfig = convTran_15len
# convTranConfig = convTran_10len
convTranConfig = convTran_5len

# convTranConfig = convTran_1_12kernels_25len
# convTranConfig = convTran_1_12kernels_20len
# convTranConfig = convTran_1_12kernels_15len
# convTranConfig = convTran_1_12kernels_10len
# convTranConfig = convTran_1_12kernels_5len
################################################################################
# ConvMamba
################################################################################
convMambaDefault: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    # lr=1e-4,
    channel_expand=2,
    seq_len=30,
    dropout =0.2,
    # dropout=0.3,
    # dropout=0.4,
    # dropout=0.5,
    # dropout=0.6,
    # dropout=0.7,
    # dropout=0.8,
    # dropout=0.9,
    # dropout=0.95,

    # lr=1e-3,
    # lr=7.5e-4,
    # lr=5e-4,
    # lr=2.5e-4,

    # lr=7.5e-5,
    # lr=5e-5,
    # lr=2.5e-5,
    lr=1e-5,
    ) 

convMamba_7emb: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=7,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
    ) 

convMamba_6emb: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=6,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
    ) 

convMamba_5emb: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=5,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
    ) 

convMamba_3emb: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=3,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
    ) 

convMamba_2emb: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=2,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
    ) 

convMamba_1emb: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=1,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
    ) 

convMamba_1ef: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
    dropout=0.2,
    expand_factor=1,
    )

convMamba_1channel: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=1,
    seq_len=30,
    # dropout=0.2,

    # dropout=0.25,
    # dropout=0.3,
    # dropout=0.35,

    # dropout=0.4,
    # dropout=0.45,
    # dropout=0.5,

    # dropout=0.55,
    # dropout=0.6,
    # dropout=0.65,

    # dropout=0.7,
    # dropout=0.75,
    # dropout=0.8,

    # dropout=0.85,
    # dropout=0.9,
    dropout=0.95,

    # expand_factor=1,
    ) 

convMamba_1ef_1channel: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=1,
    seq_len=30,
    dropout=0.2,
    expand_factor=1,
    ) 

convMamba_1ef_1channel_3emb: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=3,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=1,
    seq_len=30,
    dropout=0.2,
    expand_factor=1,
    ) 

convMamba_1ef_1channel_2emb: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=2,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=1,
    seq_len=30,
    dropout=0.2,
    expand_factor=1,
    ) 

convMamba_1ef_1channel_1emb: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=1,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=1,
    seq_len=30,
    dropout=0.2,
    expand_factor=1,
    ) 

convMamba_7state = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=7,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
    dropout=0.2,
    ) 

convMamba_6state = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=6,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
    dropout=0.2,
    ) 

convMamba_5state = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=5,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
    dropout=0.2,
    ) 

convMamba_3state = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=3,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
    dropout=0.2,
    ) 

convMamba_3state = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=3,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
    dropout=0.2,
    ) 

convMamba_2state = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=2,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
    dropout=0.2,
    ) 

convMamba_1state = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=1,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
    dropout=0.2,
    ) 


convMamba_25len: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=25,
    dropout=0.2,
    ) 

convMamba_20len: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=20,
    dropout=0.2,
    ) 

convMamba_15len: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=15,
    dropout=0.2,
    ) 

convMamba_10len: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=10,
    dropout=0.2,
    ) 

convMamba_5len: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=5,
    dropout=0.2,
    ) 

convMamba_1channel_25len = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=1,
    seq_len=25,
    dropout=0.2,
    # expand_factor=1,
    ) 

convMamba_1channel_20len = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=1,
    seq_len=20,
    dropout=0.2,
    # expand_factor=1,
    ) 

convMamba_1channel_15len = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=1,
    seq_len=15,
    dropout=0.2,
    # expand_factor=1,
    ) 

convMamba_1channel_10len = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=1,
    seq_len=10,
    dropout=0.2,
    # expand_factor=1,
    ) 

convMamba_1channel_5len = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=1,
    seq_len=5,
    dropout=0.2,
    # expand_factor=1,
    ) 
convMambaConfig = convMambaDefault
# convMambaConfig = convMamba_7emb
# convMambaConfig = convMamba_6emb
# convMambaConfig = convMamba_5emb

# convMambaConfig = convMamba_3emb
# convMambaConfig = convMamba_2emb
# convMambaConfig = convMamba_1emb
# convMambaConfig = convMamba_1ef
# convMambaConfig = convMamba_1channel
# convMambaConfig = convMamba_1ef_1channel

# convMambaConfig = convMamba_1ef_1channel_3emb
# convMambaConfig = convMamba_1ef_1channel_2emb
# convMambaConfig = convMamba_1ef_1channel_1emb

# convMambaConfig = convMamba_7state
# convMambaConfig = convMamba_6state
# convMambaConfig = convMamba_5state
# convMambaConfig = convMamba_3state
# convMambaConfig = convMamba_2state
# convMambaConfig = convMamba_1state

# convMambaConfig = convMamba_25len
# convMambaConfig = convMamba_20len
# convMambaConfig = convMamba_15len
# convMambaConfig = convMamba_10len
# convMambaConfig = convMamba_5len

# convMambaConfig = convMamba_1channel_25len
# convMambaConfig = convMamba_1channel_20len
# convMambaConfig = convMamba_1channel_15len
# convMambaConfig = convMamba_1channel_10len
# convMambaConfig = convMamba_1channel_5len