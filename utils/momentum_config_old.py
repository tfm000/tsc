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
    dropout=0.2,
)

iTransformerDefault2: ITransformerConfig = ITransformerConfig(
    dataset_name='momentum_fx_daily_model2',
    dim_ff = 3,
    emb_size = 10,
    seq_len=30,
    lr = 2.5*1e-4,
    dropout=0.2,
)

iTransformerConfig = iTransformerDefault
iTransformerConfig2 = iTransformerDefault2
################################################################################
# ConvTran
################################################################################
convTranDefault: ConvTranConfig = ConvTranConfig(
    dataset_name='momentum_fx_daily',
    lr=1e-4,
    dim_ff=3,
    dropout=0.2,
    seq_len=10,
    )

convTranDefault2: ConvTranConfig = ConvTranConfig(
    dataset_name='momentum_fx_daily_model2',
    lr=1e-4,
    dim_ff=3,
    dropout=0.2,
    seq_len=10,
    )

convTranConfig = convTranDefault
convTranConfig2 = convTranDefault2
################################################################################
# ConvMamba
################################################################################
# convMambaDefault: ConvMambaConfig = ConvMambaConfig(
#     dataset_name='momentum_fx_daily',
#     dim_ff = 8,
#     emb_size=4,
#     d_state=4,
#     seq_len=30,
#     e_layers=1,
#     lr=2.5*1e-4,
#     ) 

# convMambaDefault2: ConvMambaConfig = ConvMambaConfig(
#     dataset_name='momentum_fx_daily_model2',
#     dim_ff = 8,
#     emb_size=4,
#     d_state=4,
#     seq_len=30,
#     e_layers=1,
#     lr=2.5*1e-4,
#     )

convMambaDefault: ConvMambaConfig = ConvMambaConfig(
    dataset_name='momentum_fx_daily',
    dim_ff = 6,
    emb_size=3,
    d_state=3,
    seq_len=30,
    e_layers=1,
    lr=2.5*1e-4,
    ) 

convMambaDefault2: ConvMambaConfig = ConvMambaConfig(
    dataset_name='momentum_fx_daily_model2',
    dim_ff = 6,
    emb_size=3,
    d_state=3,
    seq_len=30,
    e_layers=1,
    lr=2.5*1e-4,
    )

convMambaConfig = convMambaDefault
convMambaConfig2 = convMambaDefault2