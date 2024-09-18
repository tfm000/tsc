"""Script allowing for easy evaluation / inference of trained models over the 
entire timeseries."""
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from utils.loss import mv_cross_entropy

@torch.no_grad()
def calc_loss_acc(y, pred_y, logits, loss_weights, config, loss2, train_model2):
    # Getting ground truth B * d_pred * n_classes
    gt = F.one_hot(y.type(torch.int64), num_classes=config.n_classes)

    # Evaluating loss
    loss1 = mv_cross_entropy(y=logits, y_pred=gt, d_pred=config.d_pred)
    
    if train_model2:
        loss = (loss1 * loss_weights[0]) + (loss2 * loss_weights[1])
    else:
        loss = loss1

    size = y.size()
    accuracy = np.where(y.cpu() == pred_y.cpu())[0].size / (size[0] * size[1])

    return loss, accuracy

@torch.no_grad()
def model_eval(model, dataloader, mean, std, config, device, model2=None, model2_classifier: bool = False, loss_weights = (1, 0), train_model2: bool = False):
    if torch.cuda.is_available():
        device: str = "cuda"
    else:
        device: str = "cpu"

    if not model.bayesian_eval:
        model.eval()
        if model2 is not None:
            model2.eval()

    mean = mean.to(device)
    std = std.to(device)
    
    dataloader.shuffle = False
    for x, y in dataloader:

        loss2 = 0

        # Calculating predictions
        if not dataloader.dataset.momentum:
            # non-momentum
            # Transfering data to GPU
            x = x.to(device)
            y = y.to(device)

            # Standardising
            x = (x - mean) / std
            pseudo_probs, logits = model(x.float())

        else:
            # momentum
            xlist = x
            ylist = y
            x = xlist[-1].to(device).float()
            y = ylist[-1].to(device)

            # Standardising
            x = (x - mean) / std

            logits2_: list = []
            for i in range(config.seq_len):
                if not model2_classifier:
                    xi = xlist[i].to(device)
                else:
                    xi = xlist[i+1].to(device)
                yi = ylist[i].to(device)

                # Standardising
                xi = (xi - mean) / std

                _, logitsi = model2(xi.float())

                # Getting ground truth B * d_pred * n_classes
                gti = F.one_hot(yi.type(torch.int64), num_classes=config.n_classes)
            
                # Evaluating loss
                loss2 += mv_cross_entropy(y=logitsi, y_pred=gti, d_pred=config.d_pred)

                logits2_.append(logitsi)

            # Appending model2 preds to x
            logits2: torch.Tensor = torch.concat(logits2_, dim=1)
            shape = x.shape
            prices: torch.Tensor = x[:,:,0].reshape((shape[0], shape[1], 1))
            x_momentum: torch.Tensor = torch.concat([prices, logits2], axis=2)

            # Calculating model predictions
            pseudo_probs, logits = model(x_momentum.float())

        pred_y = pseudo_probs.argmax(axis=2)

        # computing loss and accuracy
        loss, accuracy = calc_loss_acc(y=y, pred_y=pred_y, logits=logits, loss_weights=loss_weights, config=config, loss2=loss2, train_model2=train_model2)

        if 'cuda' in device:
            torch.cuda.empty_cache()

        return pred_y, pseudo_probs, y, loss, accuracy


@torch.no_grad()
def bayesian_eval(T: int, model, dataloader, mean, std, config, device, dropout: float = None, model2=None, model2_classifier: bool = False, loss_weights=(1, 0), train_model2: bool = False, dataset_name:str = ''):
    # putting model(s) into evaluation mode, with dropout on
    dropout = model.dropout if dropout is None else dropout
    model.bayesian_eval = True
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.p = dropout
            m.train()
    
    if dataloader.dataset.momentum:
        model2.bayesian_eval = True
        model2.eval()
        for m2 in model2.modules():
            if m2.__class__.__name__.startswith('Dropout'):
                m.p = dropout
                m2.train()

    # p_hats = []
    p_hat_sum = 0
    p_hat_sum_sq = 0
    for t in tqdm(range(T), dataset_name):
        #  model_eval(model=model, dataloader=dataloader, model2=model2, model2_classifier=model2_classifier)
        _, p_hat_t, y, _, _ =model_eval(model=model, dataloader=dataloader, mean=mean, std=std, config=config, device=device, model2=model2, model2_classifier=model2_classifier, loss_weights=loss_weights, train_model2=train_model2)
        # shape = p_hat_t.shape
        # breakpoint()
        p_hat_t = p_hat_t.permute(1, 2, 0)
        # p_hat_t.permute(1, 2, 0).permute(2, 0, 1)
        # p_hat_sum += p_hat_t
        # p_hats.append(p_hat_t)

        # p_hat_t = p_hat_t.squeeze()

        p_hat_sum += p_hat_t
        p_hat_sum_sq += p_hat_t ** 2

        if 'cuda' in device:
            # breakpoint()
            torch.cuda.empty_cache()

    # concatinating sampled values
    # samples = torch.concat(p_hats, axis=0)

    # bayesian probability prediction
    # bayes_pseudo_probs = samples.mean(axis=0, keepdim=True) 

    # uncertainity decomposition
    # epistemic = torch.mean(samples**2, axis=0, keepdim=True) - bayes_pseudo_probs**2
    # aleatoric = torch.mean(samples*(1-samples), axis=0, keepdim=True)
    # total_variance = epistemic + aleatoric

    bayes_pseudo_probs = p_hat_sum / T
    epistemic = (p_hat_sum_sq / T) - bayes_pseudo_probs ** 2
    aleatoric = (p_hat_sum - p_hat_sum_sq) / T
    # tv = epi + alea
    total_variance = epistemic + aleatoric
    # breakpoint()

    # reformatting dimensions to match TS
    bayes_pseudo_probs = bayes_pseudo_probs.permute(2, 0, 1).squeeze()
    epistemic = epistemic.permute(2, 0, 1).squeeze()
    aleatoric = aleatoric.permute(2, 0, 1).squeeze()
    total_variance = total_variance.permute(2, 0, 1).squeeze()

    # bayesian classification prediction
    bayes_pred_y = bayes_pseudo_probs.argmax(axis=1, keepdim=True)
    
    # adding predicted uncertainty to uncertainty tensors
    epistemic = torch.concat([epistemic, torch.take_along_dim(epistemic, bayes_pred_y, dim=1)], axis=1)
    aleatoric = torch.concat([aleatoric, torch.take_along_dim(aleatoric, bayes_pred_y, dim=1)], axis=1)
    total_variance = torch.concat([total_variance, torch.take_along_dim(total_variance, bayes_pred_y, dim=1)], axis=1)
    
    # calculating accuracy
    # breakpoint()
    gt = F.one_hot(y.type(torch.int64), num_classes=config.n_classes)
    size = y.size()
    accuracy = np.where(y.cpu() == bayes_pred_y.cpu())[0].size / (size[0] * size[1])

    return bayes_pred_y, bayes_pseudo_probs, y, accuracy, epistemic, aleatoric, total_variance

        