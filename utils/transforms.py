import torch
import pandas as pd


def ford_transform(df: pd.DataFrame):
    cols = df.columns.to_list()
    cols.remove('label')
    cols.remove('series')
    cols.remove('timestamp')
    cols.append('label')
    return torch.Tensor(df[cols].to_numpy())
