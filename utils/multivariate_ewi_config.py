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
)

iTransformer_4heads: ITransformerConfig = ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads = 4,
)

iTransformer_2heads: ITransformerConfig = ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads = 2,
)

iTransformer_4heads_1e: ITransformerConfig = ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads = 4,
    e_layers=1,
)

iTransformer_2heads_1e: ITransformerConfig = ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads = 2,
    e_layers=1,
)

iTransformer_1heads_1e: ITransformerConfig = ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads = 1,
    e_layers=1,
)

iTransformer_1heads_1e_5emb: ITransformerConfig = ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 5,
    seq_len=30,
    lr = 2.5*1e-4,
    # lr = 5*1e-4,
    dropout=0.2,
    num_heads = 1,
    e_layers=1,
)

iTransformer_1heads_1e_8emb: ITransformerConfig = ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 8,
    seq_len=30,
    lr = 2.5*1e-4,
    # lr = 5*1e-4,
    dropout=0.2,
    num_heads = 1,
    e_layers=1,
)

iTransformer_1heads_1e_6emb: ITransformerConfig = ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 6,
    seq_len=30,
    lr = 2.5*1e-4,
    # lr = 5*1e-4,
    # lr = 7.5*1e-4,
    dropout=0.2,
    num_heads = 1,
    e_layers=1,
)


iTransformer_1heads_1e_7emb: ITransformerConfig = ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 7,
    seq_len=30,
    lr = 2.5*1e-4,
    # lr = 5*1e-4,
    dropout=0.2,
    num_heads = 1,
    e_layers=1,
)

iTransformer_1heads_1e_4emb: ITransformerConfig = ITransformerConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size = 4,
    seq_len=30,
    lr = 2.5*1e-4,
    # lr = 5*1e-4,
    # lr = 7.5*1e-4,
    # dropout=0.2,
    # dropout=0.0,
    # dropout=0.15,
    dropout=0.1,
    num_heads = 1,
    e_layers=1,
)


# iTransformerConfig = iTransformerDefault
# iTransformerConfig = iTransformer_4heads
# iTransformerConfig = iTransformer_2heads
# iTransformerConfig = iTransformer_4heads_1e
# iTransformerConfig = iTransformer_2heads_1e

# iTransformerConfig = iTransformer_1heads_1e
# iTransformerConfig = iTransformer_1heads_1e_5emb
# iTransformerConfig = iTransformer_1heads_1e_8emb
# iTransformerConfig = iTransformer_1heads_1e_6emb

# iTransformerConfig = iTransformer_1heads_1e_7emb
iTransformerConfig = iTransformer_1heads_1e_4emb
################################################################################
# ConvTran
################################################################################
convTranDefault: ConvTranConfig = ConvTranConfig(
    dataset_name=dataset,
    lr=1e-4,
    dim_ff=3,
    dropout=0.2,
    seq_len=10,
    )

convTran_4heads: ConvTranConfig = ConvTranConfig(
    dataset_name=dataset,
    lr=1e-4,
    dim_ff=3,
    dropout=0.2,
    seq_len=10,
    num_heads=4,
    # kernel_sizes=(1, 4),
    )

convTran_1heads: ConvTranConfig = ConvTranConfig(
    dataset_name=dataset,
    lr=1e-4,
    dim_ff=3,
    dropout=0.2,
    seq_len=10,
    num_heads=1,
    )

# convTranConfig = convTranDefault
convTranConfig = convTran_4heads
# convTranConfig = convTran_1heads
################################################################################
# ConvMamba
################################################################################
convMambaDefault: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 8,
    emb_size=4,
    d_state=4,
    seq_len=30,
    e_layers=1,
    lr=2.5*1e-4,
    ) 


convMamba_1ef: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 8,
    emb_size=4,
    d_state=4,
    seq_len=30,
    e_layers=1,
    lr=2.5*1e-4,
    expand_factor=1,
    # dropout=0.4,
    # dropout=0.35,
    # dropout=0.3,
    # dropout=0.25
    ) 

convMamba_3ff_1ef: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    seq_len=30,
    e_layers=1,
    lr=2.5*1e-4,
    expand_factor=1,
    # dropout=0.4,
    # dropout=0.35,
    # dropout=0.3,
    # dropout=0.25
    ) 

convMamba_3ff_1ef_2channel: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    seq_len=30,
    e_layers=1,
    lr=2.5*1e-4,
    expand_factor=1,
    channel_expand=2,
    # dropout=0.4,
    # dropout=0.35,
    # dropout=0.3,
    # dropout=0.25
    ) 

convMamba_3ff_1ef_1channel: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    seq_len=30,
    e_layers=1,
    lr=2.5*1e-4,
    expand_factor=1,
    channel_expand=1,
    # dropout=0.4,
    # dropout=0.35,
    # dropout=0.3,
    # dropout=0.25
    ) 

convMamba_3ff_3emb_3state_1ef_1channel: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=3,
    d_state=3,
    seq_len=30,
    e_layers=1,
    lr=2.5*1e-4,
    expand_factor=1,
    channel_expand=1,
    # dropout=0.4,
    # dropout=0.35,
    # dropout=0.3,
    # dropout=0.25
    ) 

convMamba_3ff_2emb_2state_1ef_1channel: ConvMambaConfig = ConvMambaConfig(
    dataset_name=dataset,
    dim_ff = 3,
    emb_size=2,
    d_state=2,
    seq_len=30,
    e_layers=1,
    lr=2.5*1e-4,
    expand_factor=1,
    channel_expand=1,
    # dropout=0.4,
    # dropout=0.35,
    # dropout=0.3,
    # dropout=0.25
    ) 
# convMambaConfig = convMambaDefault
# convMambaConfig = convMamba_1ef
# convMambaConfig = convMamba_3ff_1ef
# convMambaConfig = convMamba_3ff_1ef_2channel
# convMambaConfig = convMamba_3ff_1ef_1channel
# convMambaConfig = convMamba_3ff_3emb_3state_1ef_1channel
convMambaConfig = convMamba_3ff_2emb_2state_1ef_1channel