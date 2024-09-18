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
from utils.training import train_epoch, SaveBestModel, load_model, plotter


if __name__ == '__main__':
    ############################################################################
    # User Inputs
    ############################################################################
    # Training specs
    # training: str = 'convtran'  # done
    # training: str = 'convmamba'  # done
    training: str = 'itransformer'  # done
    batch_size: int = 50
    num_epochs = 100
    min_epochs: int = 30
    dataset_split: tuple = (0.7, 0.2, 0.1)
    split_indices: tuple = (0, 6118, 7910)
    dataset_burnins: tuple = (0, 25, 25) # needs to increase for momentum - done below
    save_weights = (7/9, 2/9, 0)
    # final_model: bool = True
    momentum: bool = True
    # momentum: bool = False

    # pretrain_model2: bool = False
    pretrain_model2: bool = True
    
    train_model2: bool = True
    # train_model2: bool = False

    # model2_classifier: bool = False
    model2_classifier: bool = True
    
    loss_weights = (0.6, 0.4)
    # dataset_path: str = './data/eurusd_univariate.xlsx'
    dataset_path: str = './data/eurusd_multivariate_ewi_ft.xlsx'
    # prob_model_type: str = 'univariate'
    prob_model_type: str = 'multivariate'
    best_prior_epoch_value = np.inf

    # if not final_model:
    #     save_folder: str = f'./runs/eurusd_momentum/{training}'
    # else:
    #     save_folder: str = f'./runs/eurusd_momentum/final_models/'
    save_folder: str = f'./runs/eurusd_momentum/{training}'

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
    if training == 'convtran':
        # convtran imports
        from utils.momentum_config import convTranConfig as config
        from utils.finetuned_configs import convTran_univariate as config2
        from convtran.Models.model import ConvTran as Model
        # run_identifier: str = f'{config.seq_len}seq + {config.dim_ff}ff + {config.emb_size}emb + {config.channel_expand}channel+ {config.num_heads}heads + {str(config.kernel_sizes)}kernels + {config.dropout}dropout + {config.lr}lr'
    elif training == 'convmamba':
        # ConvMamba imports
        from utils.momentum_config import convMambaConfig as config
        from utils.finetuned_configs import convMamba_univariate as config2
        from mamba.models.convmamba import ConvMamba as Model
        # run_identifier: str = f'{config.seq_len}seq + {config.dim_ff}ff + {config.emb_size}emb + {config.d_state}state + {config.e_layers}e+ + {config.expand_factor}ef + {config.channel_expand}channel + {config.dropout}dropout + {config.lr}lr'
        # pretrain_path: str = f'C:\Projects\tsc\runs\eurusd_univariate\convmamba\{run_identifier}'
    elif training == 'itransformer':
        # iTransformer imports
        from utils.momentum_config import iTransformerConfig as config

        if prob_model_type == 'univariate':
            from utils.finetuned_configs import iTransformer_univariate as config2
        else:
            from utils.finetuned_configs import iTransformer_multivariate as config2        
        from itransformer.models.iTransformer import Model
        # run_identifier: str = f'{config.seq_len}seq + {config.dim_ff}ff + {config.emb_size}emb + {config.num_heads}heads + {config.e_layers}e + {config.dropout}dropout + {config.lr}lr'
    
    run_identifier: str = prob_model_type
    if pretrain_model2:
        ext = '_ewi_ft' if prob_model_type == 'multivariate' else ''
        pretrain_path: str = f'C:\\Projects\\tsc\\runs\eurusd_{prob_model_type}{ext}\\final_models\\{training} - {prob_model_type}.pt'
        run_identifier = f"pretrained + {run_identifier}"

    if model2_classifier:
        run_identifier = f'classifier + {run_identifier}'

    if train_model2:
        run_identifier = f'trainable + {run_identifier}'    
    else:
        run_identifier = f'non trainable + {run_identifier}' 

    # if final_model:
    #     run_identifier: str = 'multivariate'

    seq_len: int = config.seq_len
    n_classes: int = config.n_classes
    
    # Defining model
    model = Model(config).float()
    model = model.to(device)

    if momentum:
        model2 = Model(config2).float()
        model2 = model2.to(device)

        dataset_burnins = (dataset_burnins[0], dataset_burnins[1] + config.seq_len, dataset_burnins[2] + config.seq_len)
    else:
        model2 = None

    # Printing number of params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nparams = sum([np.prod(p.size()) for p in model_parameters])
    
    if momentum:
        model2_parameters = filter(lambda p: p.requires_grad, model2.parameters())
        nparams2 = sum([np.prod(p.size()) for p in model2_parameters])
        s = f"Training {training.capitalize()} with m1 {nparams} parameters and m2 {nparams2} parameters."
    else:
        s = f"Training {training.capitalize()} with {nparams} parameters."
    print(s)

    # Loading pretrained model
    if pretrain_model2:
        model2 = load_model(model=model2, path=pretrain_path, device=device)
        model2.train()

    # Defining optimiser
    if momentum:
        parameters = list(model.parameters()) + list(model2.parameters())
    else:
        parameters = model.parameters()
    optim = config.optimiser(parameters, lr=config.lr, weight_decay=config.weight_decay)

    # Getting data
    dataloaders = get_dataloaders(root=dataset_path, model_config=config, batch_size=batch_size, shuffle=True, burnins=dataset_burnins, split=dataset_split, split_indices=split_indices, transform=None, index_col=0, momentum=momentum)
    
    # Setting run-name
    run_name: str = f"{training} - {run_identifier}"

    # Defining saver
    save_path: str = f"{save_folder}\\{run_name}"
    saver: SaveBestModel = SaveBestModel(save_path=f"{save_path}.pt", save_weights=save_weights, min_epochs=min_epochs, best_prior_epoch_value=best_prior_epoch_value, save_path2=f"{save_path}_model2.pt")

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

        res = train_epoch(model=model, dataloaders=dataloaders, config=config, epoch=epoch, optim=optim, show_test=False, device=device, model2=model2, momentum=momentum, loss_weights=loss_weights, train_model2=train_model2, model2_classifier=model2_classifier)
        
        if momentum:
            model, epoch_record, model2 = res
        else:
            model, epoch_record = res

        # saving history
        records.append(epoch_record)

        # saving model
        saver(model=model, epoch_record=epoch_record, model2=model2)
    
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
        
        