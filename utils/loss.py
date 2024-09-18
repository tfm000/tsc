import torch.nn.functional as F
import torch


def mv_cross_entropy(y: torch.Tensor, y_pred: torch.Tensor, d_pred: int) -> float:
    y_pred = y_pred.float()
    cross_entropy: torch.Tensor = torch.FloatTensor([0.0]).to(y.device)
    for i in range(d_pred):
        cross_entropy += F.cross_entropy(y[:, i, :], y_pred[:, i, :])
    return cross_entropy
        
