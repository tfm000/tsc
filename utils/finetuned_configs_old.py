from utils.config import ConvMambaConfig, ITransformerConfig, ConvTranConfig

__all__ = ['convMamba_univariate', 'iTransformer_univariate', 'convTran_univariate']

convMamba_univariate: ConvMambaConfig = ConvMambaConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 8,
    emb_size=4,
    d_state=4,
    dropout=0.125,
    lr=1e-4,
) # 2.1k params

iTransformer_univariate: ITransformerConfig = ITransformerConfig(
    dataset_name='univariate_fx_daily',
    dim_ff = 10,
    emb_size = 10,
    lr = 1e-4,
    dropout = 0.125,
) # 2.3k params

convTran_univariate: ConvTranConfig = ConvTranConfig(
    dataset_name='univariate_fx_daily',
    lr=1e-4,
    dim_ff=2,
    dropout=0.1,
    ) # 4.3k params

