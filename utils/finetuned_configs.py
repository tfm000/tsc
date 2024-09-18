from utils.config import ConvMambaConfig, ITransformerConfig, ConvTranConfig, MambaConfig

# __all__ = ['convMamba_univariate', 'iTransformer_univariate', 'convTran_univariate', 'mamba_univariate']

iTransformer_univariate: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0,
    num_heads=1,
    e_layers=1,
) # 916 params


convTran_univariate: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
    ) # 1109 params


convMamba_univariate: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
) # 806 params


mamba_resnet_univariate_underfit: MambaConfig = MambaConfig(
    dataset_name='univariate_fx_daily',
    residuals=True,
    seq_len=30
) # 1877 params

################################################################################
# Multivariate
################################################################################
iTransformer_multivariate: ITransformerConfig = ITransformerConfig(
    dataset_name='multivariate_fx_daily',
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
    num_heads=1,
    e_layers=1,
) # 1036 params


convTran_multivariate_overfit: ConvTranConfig = ConvTranConfig(
    dataset_name='multivariate_fx_daily',
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
) # 1621 params


convMamba_multivariate_overfit: ConvMambaConfig = ConvMambaConfig(
    dataset_name='multivariate_fx_daily',
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
) # 934 params

################################################################################
# Momentum
################################################################################
iTransformer_momentum: ITransformerConfig = ITransformerConfig(
    dataset_name='momentum_fx_daily',
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0,
    num_heads=1,
    e_layers=1,
)