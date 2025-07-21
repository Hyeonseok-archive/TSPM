import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def seed_everything(seed:int=42, deterministic=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])
    

# preprocessing
def make_sequence(data, seq_len: int = 24*7):
    total_len = seq_len * 2

    seqeunces = [
        data[i:(i + total_len)]
        for i in range(len(data) - total_len + 1)
    ]

    X = np.array([seqeunce[:seq_len]
                 for seqeunce in seqeunces], dtype='float32')
    Y = np.array([seqeunce[seq_len:]
                 for seqeunce in seqeunces], dtype='float32')

    return X, Y


def standard(arr, axis=None, mean=None, std=None, inverse=False):
    if inverse:
        return arr * std + mean
    else:
        if axis is None:
            mean_ = arr.mean()
            std_ = arr.std()
        else:
            mean_ = arr.mean(axis=axis)
            std_ = arr.std(axis=axis)

        div = np.where(std_ == 0, 1, std_)
        # arr_ = (arr - mean_) / std_
        arr_ = (arr - mean_) / div
        return arr_, mean_, std_


def minmax(arr, axis=None, min_val=None, max_val=None, inverse=False):
    if inverse:
        return arr * (max_val - min_val) + min_val
    else:
        if axis is None:
            min_ = arr.min()
            max_ = arr.max()
        else:
            min_ = arr.min(axis=axis)
            max_ = arr.max(axis=axis)
        div = (max_ - min_)
        div = np.where(div == 0, 1, div)
        # arr_ = (arr - min_) / (max_ - min_)
        arr_ = (arr - min_) / div
        return arr_, min_, max_
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# metrics
def MSE(pred, true, axis=None, root=False):
    if axis is None:
        if root:
            return np.sqrt(np.mean((pred - true)**2))
        else:
            return np.mean((pred - true)**2)
    else:
        if root:
            return np.sqrt(np.mean((pred - true)**2, axis=axis))
        else:
            return np.mean((pred - true)**2, axis=axis)
        

def MAE(pred, true, axis=None):
    if axis is None:
        return np.mean(np.abs(pred - true))
    else:
        return np.mean(np.abs(pred - true), axis=axis)
    

def MAPE(pred, true, axis=None):
    # for escaping from 0 division error, add 1 value
    pred += 1
    true += 1
    if axis is None:
        return np.mean(np.abs((pred - true) / true)) * 100
    else:
        return np.mean(np.abs((pred - true) / true), axis=axis) * 100
    

def SMAPE(pred, true, axis=None):
    # for escaping from 0 division error, add 1 value
    pred += 1
    true += 1
    numerator = np.abs(pred - true)
    denominator = (np.abs(pred) + np.abs(true)) / 2
    if axis is not None:
        return np.mean(numerator / denominator, axis=axis) * 100
    else:
        return np.mean(numerator / denominator) * 100