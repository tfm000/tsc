from torch.utils.data import Dataset, DataLoader
from typing import Callable
import torch
import pandas as pd
from pathlib import Path
import os


class ClassificationDataset(Dataset):
    """Class for loading and sampling a classification dataset."""

    def __init__(self, root: str, seq_len: int, transform: Callable=None, index_col: int=None) -> None:
        """Class for loading and sampling a classification dataset."""
        self.seq_len: int = seq_len

        # getting data
        df_raw: pd.DataFrame = pd.read_csv(root, index_col=index_col)
        
        # transforming data
        # transformation must make the last col the labels
        transform = (lambda x: torch.tensor(x.to_numpy())) if transform is None else transform
        self.data: torch.Tensor = transform(df_raw)

        # checking valid
        self.T, N = self.data.shape
        if self.T - self.seq_len < 0:
            raise ValueError(f'Provided data sequence is too small for specified sequence length: {self.T} < {self.seq_len}')
        
    def __len__(self) -> int:
        return self.T - self.seq_len
    
    def __getitem__(self, index) -> torch.Tensor:
        sample: torch.Tensor = self.data[index:index+self.seq_len]
        return sample[:, :-1], sample[-1, -1]


class MultiVariateDataset(Dataset):

    def __init__(self, root: str, model_config, transform: Callable=None, 
                 burnin: int=0, index_col: int=None, momentum: bool=False) -> None:
        """Class for loading and sampling a dataset for training an ML model. 
        This dataset and model must perform either TSF or TSC, with labels 
        predicted for each variable in the TS.
        
        Note this object does not split the data into train, val and test sets; 
        use multiple dataset objects for this.

        Parameters
        ----------
        root: str
            The location of the excel file containing the dataset series and 
            labels. The series itself must be within a tab named 'series'
            and the labels within a tab called 'labels'.
        seq_len: int
            The length of the TS data samples to generate when training.
        transform: Callable
            A callable function to apply to the entire dataset series.
        index_col: int | None
            The column to use as an index for both dataset series and labels.
        burnin: int
            The initial datapoints of the dataset to discard. Useful for 
            preventing information leakage between the training, validation 
            and test sets.
        """
        self.seq_len: int = model_config.seq_len
        self.momentum: bool = momentum
        self.buffer = self.seq_len * 2 if self.momentum else self.seq_len

        # getting data
        excel: pd.ExcelFile = pd.ExcelFile(root)
        df_raw: pd.DataFrame = excel.parse(sheet_name='series', index_col=0).iloc[burnin:]
        
        # breakpoint()
        # df_raw = df_raw.drop(columns=['VIX'])

        labels_raw: pd.DataFrame = excel.parse(sheet_name='labels', index_col=0).iloc[burnin:]
        self.T: int = len(df_raw)
        self.df_raw = df_raw
        self.labels_raw = labels_raw

        # checking valid
        d_model = model_config.d_model if 'nvars' not in dir(model_config) else model_config.nvars
        d_model = d_model - model_config.n_classes if self.momentum else d_model
        # if len(df_raw.columns) != d_model:
        #     # breakpoint()
        #     raise ValueError("Number of variables mismatch between data series and model.")
        # if len(labels_raw.columns) != model_config.d_pred:
        #     raise ValueError("Number of variables mismatch between labels and model output.")
        # if len(df_raw) != len(labels_raw):
        #     raise ValueError("Length mismatch between data series and labels.")
        # if self.T - self.seq_len < 0:
        #     raise ValueError(f'Provided data sequence is too small for specified sequence length: {self.T} < {self.seq_len}')
        
        # transforming data
        transform = (lambda x: torch.tensor(x.to_numpy())) if transform is None else transform
        self.data: torch.Tensor = transform(df_raw)
        self.labels: torch.Tensor = torch.IntTensor(labels_raw.to_numpy())

        # self.data = torch.flip(self.data, (0,1))
        # self.labels=torch.flip(self.labels, (0,1))
        # breakpoint()

    def __len__(self) -> int:
        return self.T - self.buffer + 1

    def __getitem__(self, index) -> torch.Tensor:
        if not self.momentum:
            sample: torch.Tensor = self.data[index:index+self.seq_len]
            sample_label: torch.Tensor = self.labels[index+self.seq_len-1]
            return sample, sample_label

        samples = []
        sample_labels = []
        
        # timestamps = []
        # mean = torch.load(r'C:\Projects\tsc\runs\eurusd_momentum\itransformer\mean.pt')
        # std = torch.load(r'C:\Projects\tsc\runs\eurusd_momentum\itransformer\std.pt') 
        # for i in range(self.seq_len):
        #     sample: torch.Tensor = (self.data[index + i: index + self.seq_len + i] * std) + mean
        #     sample_label: torch.Tensor = self.labels[index + self.seq_len - 1 + i]
        #     samples.append(sample)
        #     sample_labels.append(sample_label)
        #     ts = str(self.df_raw.iloc[index + self.seq_len - 1 + i].name)
        #     # breakpoint()
        #     timestamps.append(ts)
        # return samples, sample_labels, timestamps

        for i in range(self.seq_len+1):
            sample: torch.Tensor = self.data[index + i: index + self.seq_len + i]
            sample_label: torch.Tensor = self.labels[index + self.seq_len - 1 + i]
            samples.append(sample)
            sample_labels.append(sample_label)
        return samples, sample_labels
            

def create_datasets(root: str, split: tuple = None, index_col: int=None, split_indices: tuple = None) -> None:
    cfolder = os.path.dirname(root)
    excel_name = f"{root.replace(cfolder, '').replace('.xlsx', '')}.xlsx"
    nfolder = f"{cfolder}{excel_name.replace('.xlsx', '')}"

    Path(nfolder).mkdir(parents=True, exist_ok=True)

    excel = pd.ExcelFile(f'{cfolder}{excel_name}', 'openpyxl')
    size = len(excel.parse('series'))

    if split_indices is None and split is not None:
        train, val = int(split[0]*size), int(split[1]*size)
        split_indices = (0, train, val+train, size)
    elif split_indices is not None:
        split_indices = (*split_indices[:3], size)
    elif split_indices is None and split is None:
        raise ValueError('one of split or split_indices must be passed.')
    
    sheets = ('series', 'labels')
    for i, s in enumerate(('train', 'validation', 'test')):
        with pd.ExcelWriter(f'{nfolder}/{s}.xlsx') as writer:
            for sheet in sheets:
                df: pd.DataFrame = excel.parse(sheet_name=sheet, index_col=index_col)
                sub_df: pd.DataFrame = df.iloc[split_indices[i]:split_indices[i+1]]
                # breakpoint()
                sub_df.to_excel(writer, sheet_name=sheet)


def get_dataloaders(root: str, model_config, batch_size: int, shuffle: bool, burnins: tuple = (0, 0, 0), split: tuple = None, split_indices = None, strategy_eval: bool = False, transform: Callable=None, index_col: int=None, momentum: bool = False, incl_combined: bool = False) -> tuple:
    root = root.replace('.xlsx', '')
    if not os.path.exists(root):
        assert split is not None, 'split cannot be none if individual train, validation and test excels do not exist.'
        create_datasets(root=root, split=split, index_col=index_col, split_indices=split_indices)
    
    dataloaders = []
    dataset_names: bool = (r'/train', r'/validation', r'/test')
    if incl_combined:
        burnins = (*burnins, 0)
        dataset_names = (*dataset_names, '')
    for i, name in enumerate(dataset_names):
        dataset = MultiVariateDataset(root=f"{root}{name}.xlsx", model_config=model_config, transform=transform, burnin=burnins[i], index_col=index_col, momentum=momentum)
        if ('train' not in name) or strategy_eval:
            batch_size = len(dataset)
            shuffle = False
        # breakpoint()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        dataloaders.append(dataloader)
    return tuple(dataloaders)

        



    