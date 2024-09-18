from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from utils.loss import mv_cross_entropy


def evaluation_loop(model, dataloader, update: bool, config, epoch: int, device: str, model2, momentum: bool, loss_weights: bool, train_model2: bool, model2_classifier: bool, optim = None) -> tuple:
    # model2 is the smaller mv model.
    epoch_loss = 0
    epoch_accuracy = 0
    c = 0

    # Zeroing gradient
    if update:
        optim.zero_grad()
        model.train()
        
        if momentum:
            if train_model2:
                model2.train() 
            else:
                model2.eval() # dropout turned off
    else:
         model.eval()

         if momentum:
            model2.eval()

    iterator = tqdm(dataloader, f'epoch {epoch}') if update else dataloader
    # breakpoint()
    # for x, y, timestamps in iterator:
    for x, y in iterator:
        # Transfering data to GPU

        if not momentum:
            # breakpoint()
            x = x.to(device).float()
            y = y.to(device)

            # Calculating predictions
            pseudo_probs, logits = model(x)

            # Getting ground truth B * d_pred * n_classes
            gt = F.one_hot(y.type(torch.int64), num_classes=config.n_classes)
            
            # Evaluating loss
            loss = mv_cross_entropy(y=logits, y_pred=gt, d_pred=config.d_pred)
            # loss = F.cross_entropy(logits, gt.float())

            if update:
                # Gradient step
                loss.backward()
                optim.step()
                optim.zero_grad()
            if device == 'cuda:0':
                torch.cuda.empty_cache()
        
        else:
            # Adjusting inputs
            xlist = x
            ylist = y
            x = xlist[-1].to(device).float()
            # x = xlist.pop(-1).to(device).float()
            y = ylist.pop(-1).to(device)
            # timestamp = timestamps.pop(-1)

            # Calculating model2 predictions
            loss2 = 0
            logits2_: list = []
            # xi=9
            # breakpoint()
            for i in range(config.seq_len):
                # xi_=xi
                if not model2_classifier:
                    xi = xlist[i].to(device).float()
                else:
                    xi = xlist[i+1].to(device).float()
                    # breakpoint()
                yi = ylist[i].to(device)
                # timestampi = timestamps[i]
                
                # import numpy as np
                # np.array(timestamps)[:, 0]

                # breakpoint()
                
                _, logitsi = model2(xi)

                # Getting ground truth B * d_pred * n_classes
                gti = F.one_hot(yi.type(torch.int64), num_classes=config.n_classes)
            
                # Evaluating loss
                loss2 += mv_cross_entropy(y=logitsi, y_pred=gti, d_pred=config.d_pred)

                logits2_.append(logitsi)
            loss2 /= config.seq_len
            

            # Appending model2 preds to x
            logits2: torch.Tensor = torch.concat(logits2_, dim=1)
            # breakpoint()
            shape = x.shape
            prices: torch.Tensor = x[:,:,0].reshape((shape[0], shape[1], 1))
            x_momentum: torch.Tensor = torch.concat([prices, logits2], axis=2)

            # Calculating model predictions
            pseudo_probs, logits = model(x_momentum)

            # Getting ground truth B * d_pred * n_classes
            gt = F.one_hot(y.type(torch.int64), num_classes=config.n_classes)
            
            # Evaluating loss
            loss1 = mv_cross_entropy(y=logits, y_pred=gt, d_pred=config.d_pred)
            
            if train_model2:
                loss = (loss1 * loss_weights[0]) + (loss2 * loss_weights[1])
            else:
                loss = loss1

            if update:
                # Gradient step
                loss.backward()
                optim.step()
                optim.zero_grad()
            if device == 'cuda:0':
                torch.cuda.empty_cache()

        # Recording performance
        epoch_loss += loss
        pred_y = pseudo_probs.argmax(axis=2)
        size = y.size()
        epoch_accuracy += np.where(y.cpu() == pred_y.cpu())[0].size / (size[0] * size[1])
        c+=1

        # if epoch == 99:
        #     x_full = torch.load('./runs/eurusd_univariate_ds/x.pt')
        #     y_full = torch.load('./runs/eurusd_univariate_ds/y.pt')
        #     breakpoint()

    epoch_loss /= c
    epoch_accuracy /= c

    return model, float(epoch_loss.detach()), epoch_accuracy, model2


def train_epoch(model, dataloaders: tuple, config, epoch: int, optim, device: str, show_test: bool=False, model2 = None, momentum: bool = False, loss_weights: bool = (0.0, 1.0), train_model2: bool = True, model2_classifier: bool = False) -> tuple:
    # training
    model, train_loss, train_acc, model2 = evaluation_loop(train_model2=train_model2, model2_classifier=model2_classifier, model=model, dataloader=dataloaders[0], update=True, config=config, epoch=epoch, optim=optim, device=device, model2=model2, momentum=momentum, loss_weights=loss_weights)

    # validation
    _, val_loss, val_acc, _ = evaluation_loop(train_model2=train_model2, model2_classifier=model2_classifier, model=model, dataloader=dataloaders[1], update=False, config=config, epoch=epoch, device=device, model2=model2, momentum=momentum, loss_weights=loss_weights)
    
    # test
    _, test_loss, test_acc, _ = evaluation_loop(train_model2=train_model2, model2_classifier=model2_classifier, model=model, dataloader=dataloaders[2], update=False, config=config, epoch=epoch, device=device, model2=model2, momentum=momentum, loss_weights=loss_weights)
    
    # recording values
    loss: pd.DataFrame = pd.DataFrame({'Train': [train_loss], 'Val': [val_loss], 'Test': [test_loss]}, index=[epoch])
    accuracy: pd.DataFrame = pd.DataFrame({'Train': [train_acc], 'Val': [val_acc], 'Test': [test_acc]}, index=[epoch])
    record: pd.DataFrame = pd.concat([loss, accuracy], axis=1, keys=['Loss', 'Accuracy'])
    
    # printing info
    print_record: pd.DataFrame = record if show_test else record.drop(columns=[('Loss', 'Test'), ('Accuracy', 'Test')])
    print(print_record)

    if momentum:
        return model, record, model2
    return model, record


class SaveBestModel:
    
    def __init__(self, save_path: str, save_weights: tuple, min_epochs: int, metric: str = 'Accuracy', save_path2: str = None, best_prior_epoch_value: float = None):
        # location to save too
        self.save_path: str = save_path
        self.save_path2: str = save_path2
          
        # dataset and metric to evaluate the 'best' model
        self.save_weights: tuple = save_weights
        self.metric: str = metric.capitalize()

        # minimum number of epochs which must pass
        self.min_epochs: int = min_epochs

        # initialising
        best_prior_epoch_value = np.inf if best_prior_epoch_value is None else best_prior_epoch_value
        self.best_value: float = best_prior_epoch_value
        self.best_epoch: int = -np.inf
        self.improvement: bool = False
    
    def __call__(self, model, epoch_record: pd.DataFrame, model2=None):
        epoch_number: int = int(epoch_record.index[0])
        if epoch_number < self.min_epochs:
            return None

        epoch_value: float = (epoch_record[self.metric] * self.save_weights).sum(axis=1).values[0]
        if self.metric == 'Accuracy':
            epoch_value = -epoch_value
        if epoch_value < self.best_value:
            self.improvement = True
            self.best_value = epoch_value
            self.best_epoch = int(epoch_record.index[0]) 
            torch.save(model.state_dict(), self.save_path)
            if model2 is not None:
                torch.save(model2.state_dict(), self.save_path2)
            # torch.save(model, self.save_path)


def load_model(model, path: str, device: str = 'cpu'):
    # state_dict = torch.load(path, map_location=device, weights_only=True)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

def plotter(history: pd.DataFrame, saver: SaveBestModel, show: bool) -> None:
    best_epoch: int = saver.best_epoch
    save_path: str = saver.save_path
    eps: float = 1e-4
    xbounds: tuple = (best_epoch-eps, best_epoch+eps)
    plot_columns = ['Train', 'Val', 'Test']
    titles = ('Loss', 'Accuracy')
    legend_loc = ('upper right', 'lower right')
    for i in range(2):
        title, loc = titles[i], legend_loc[i]
        data: pd.DataFrame = history[title][plot_columns]
        ybounds = data.min().min(), data.max().max()
        data.plot()
        plt.plot(xbounds, ybounds, zorder=0, c='black', label='Saved Model')
        plt.legend(loc=loc)
        plt.title(title)
        plt.grid()

        if saver.improvement:
            plt.savefig(f'{save_path}-{title}.png')
    
    if show:
        plt.show()

