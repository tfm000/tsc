"""Script for training models.

Instructions
------------
1. Adjust the variable values in user inputs to your liking.
2. If you wish to adjust model specific hyperparams, each model has a config 
    file located inside its subdirectory. Inside will be an object specifying 
    the hyperparameters for each model, which you can edit.

Models Implemented
------------------
- ConvTran
- iTransformer
- Bi-Mamba
- Bi-Mamba Fusion
- ConvMamba
- Mamba
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.dataloader import get_dataloaders
from utils.training import train_epoch, SaveBestModel, plotter


if __name__ == '__main__':
    ############################################################################
    # User Inputs
    ############################################################################
    # Training specs
    # training: str = 'convmamba' # done
    # training: str = 'convtran' # done
    # training: str = 'itransformer' # done
    training: str = 'mamba' # done
    batch_size: int = 50
    num_epochs = 100
    min_epochs: int = 30
    dataset_split: tuple = (0.7, 0.2, 0.1)
    dataset_burnins: tuple = (0, 25, 25)
    save_weights = (7/9, 2/9, 0)
    # final_model: bool = True
    final_model: bool = False
    dataset_path: str = './data/eurusd_univariate.xlsx'
    best_prior_epoch_value = np.inf

    if not final_model:
        save_folder: str = f'./runs/eurusd_univariate/{training}'
    else:
        save_folder: str = f'./runs/eurusd_univariate/final_models/'
    
    show: bool = True

    ############################################################################
    # Connecting to GPU
    ############################################################################
    if torch.cuda.is_available():
        device: str = "cuda"
    else:
        device: str = "cpu"
    # device = "cpu"
    print(f"Training on {device}")

    ############################################################################
    # Model Training
    ############################################################################
    # Model specific imports
    if training == 'convmamba':
        # ConvMamba imports
        if final_model:
            best_prior_epoch_value = -0.558394
            from utils.finetuned_configs import convMamba_univariate as config
        else:
            from utils.config import convMambaConfig as config
        from mamba.models.convmamba import ConvMamba as Model
        run_identifier: str = f'{config.seq_len}seq + {config.dim_ff}ff + {config.emb_size}emb + {config.d_state}state + {config.e_layers}e+ + {config.channel_expand}channel + {config.dropout}dropout + {config.lr}lr'
        # run_identifier: str = f'{batch_size}batch + {config.dim_ff}ff + {config.emb_size}emb + {config.d_state}state + {config.e_layers}e+ {config.dropout}dropout + {config.lr}lr'
    elif training == 'convtran':
        # convtran imports
        if final_model:
            best_prior_epoch_value = -0.570566
            from utils.finetuned_configs import convTran_univariate as config
        else:
            from utils.config import convTranConfig as config
        from convtran.Models.model import ConvTran as Model
        run_identifier: str = f'{config.seq_len}seq + {config.dim_ff}ff + {config.emb_size}emb + {config.channel_expand}channel+ {config.num_heads}heads + {config.dropout}dropout + {config.lr}lr'
    elif training == 'itransformer':
        # iTransformer imports
        if final_model:
            best_prior_epoch_value = -0.557008
            from utils.finetuned_configs import iTransformer_univariate as config
        else:
            from utils.config import iTransformerConfig as config
        from itransformer.models.iTransformer import Model
        run_identifier: str = f'{config.seq_len}seq + {config.dim_ff}ff + {config.emb_size}emb + {config.num_heads}heads + {config.e_layers}e + {config.dropout}dropout + {config.lr}lr'
    elif training == 'mamba':
        # Mamba imports
        if final_model:
            from utils.finetuned_configs import mamba_resnet_univariate_underfit as config
        else:
            from utils.config import mambaConfig as config
        from mamba.models.mamba import Mamba as Model
        resids = 'resids + ' if config.residuals else ''
        run_identifier: str = f'{resids}{config.seq_len}seq + {config.dim_ff}ff + {config.d_state}state + {config.e_layers}e + {config.dropout}dropout + {config.lr}lr'
    
    if final_model:
        run_identifier: str = 'univariate'

    seq_len: int = config.seq_len
    n_classes: int = config.n_classes
    
    # Defining model
    model = Model(config).float()
    model = model.to(device)

    # Printing number of params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nparams = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Training {training.capitalize()} with {nparams} parameters.")

    # Defining optimiser
    optim = config.optimiser(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Getting data
    dataloaders = get_dataloaders(root=dataset_path, model_config=config, batch_size=batch_size, shuffle=True, burnins=dataset_burnins, split=dataset_split, transform=None, index_col=0)
    
    # Setting run-name
    run_name: str = f"{training} - {run_identifier}"

    # Defining saver
    save_path: str = f"{save_folder}\\{run_name}"
    saver: SaveBestModel = SaveBestModel(save_path=f"{save_path}.pt", save_weights=save_weights, min_epochs=min_epochs, best_prior_epoch_value=best_prior_epoch_value)

    # Normalising data
    training_data: torch.Tensor = dataloaders[0].dataset.data
    mean = training_data.mean(axis=0)
    std = training_data.std(axis=0)
    for i in range(3):
        dataloaders[i].dataset.data = (dataloaders[i].dataset.data - mean) / std

    # Training
    records: list = []
    for epoch in range(num_epochs):
        if device == 'cuda:0':
            torch.cuda.empty_cache()
        model, epoch_record = train_epoch(model=model, dataloaders=dataloaders, config=config, epoch=epoch, optim=optim, show_test=False, device=device)
        
        # saving history
        records.append(epoch_record)

        # saving model
        saver(model=model, epoch_record=epoch_record)

    # creating and adding weighted average to history
    history: pd.DataFrame = pd.concat(records)
    categories = ['Loss', 'Accuracy']
    for category in categories:
        history[(category, 'WA')] = (history[category] * save_weights).sum(axis=1)
    history = history[categories]

    # saving
    if saver.improvement:
        # saving history
        history.to_csv(f"{save_path}.csv")

        # saving mean and std
        torch.save(mean, f'{save_folder}/mean.pt')
        torch.save(std, f'{save_folder}/std.pt')

    # printing best epoch
    print(f"\n{training} Improvement: {saver.improvement}")

    if saver.improvement:
        print("Best Epoch")
        print(history.iloc[saver.best_epoch].to_frame().T)
    
    # plotting
    plotter(history=history, saver=saver, show=show)

        