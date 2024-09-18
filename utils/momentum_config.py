from utils.config import ConvMambaConfig, ITransformerConfig, ConvTranConfig, MambaConfig

################################################################################
# iTransformer
################################################################################
iTransformerDefault: ITransformerConfig = ITransformerConfig(
    dataset_name='momentum_fx_daily',
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0,
    num_heads=1,
    e_layers=1,
)

iTransformerConfig = iTransformerDefault
################################################################################
# ConvTran
################################################################################
convTranDefault: ConvTranConfig = ConvTranConfig(
    dataset_name='momentum_fx_daily',
    dim_ff=2,
    channel_expand=2,
    emb_size=8,
    seq_len=30,
    lr=1e-3,
    dropout=0.2,
    )

convTranConfig = convTranDefault
################################################################################
# ConvMamba
################################################################################
convMambaDefault: ConvMambaConfig = ConvMambaConfig(
    dataset_name='momentum_fx_daily',
    dim_ff = 3,
    emb_size=4,
    d_state=4,
    e_layers=1,
    lr=1e-4,
    channel_expand=2,
    seq_len=30,
    ) 

convMambaConfig = convMambaDefault