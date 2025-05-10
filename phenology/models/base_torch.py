import os.path
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from termcolor import colored

import phenology.config as config

from phenology.dataset.dataset import Dataset
from phenology.models import ModelException
from phenology.models.base import BaseModel
from phenology.models.util.dataset_wrapper import TorchDataset
from phenology.models.util.early_stopping import EarlyStopper


class BaseTorchModel(BaseModel, torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__()
        super(nn.Module, self).__init__()

    def predict(self, x: dict, device=torch.device('cpu'), **kwargs: dict) -> tuple:

        with torch.no_grad():
            x = TorchDataset.cast_to_tensor(x)
            x = TorchDataset.sample_to_device(x, device)
            x = TorchDataset.collate_fn([x])

            ix, info = self(x)

            [ix] = self._ixs_to_int(ix)

            # fmt = 'h' if self._step == 'hourly' else 'D'
            fmt = 'D'
            dt = np.timedelta64(ix, fmt)

            date = x['season_start'][0] + dt

        return date, {'forward_pass': info}

    def predict_all(self, xs: list, device=torch.device('cpu'),
                    **kwargs: dict) -> list:  # TODO -- define dataloader, batch size

        raise NotImplementedError

        # with torch.no_grad():
        #     xs = [TorchDataset.cast_to_tensor(sample) for sample in xs]
        #     xs = [TorchDataset.sample_to_device(sample, device) for sample in xs]
        #     xs = TorchDataset.collate_fn(xs)
        #
        #     ix, info = self(xs)
        #
        #     ixs = self._ixs_to_int(ix)
        #
        #     doys = [index_to_doy(ix,
        #                          src=x[config.KEY_DATA_SOURCE],
        #                          species_code=x[config.KEY_SPECIES_CODE],
        #                          subgroup_code=x[config.KEY_SUBGROUP_CODE],
        #                          year=x[config.KEY_YEAR],
        #                          ) for ix, x in zip(ixs, xs)]
        #
        # return [(doy, info) for doy in doys]  # TODO -- info dict properly

    @staticmethod
    def _ixs_to_int(ixs: torch.Tensor) -> list:
        # Round all elements to the nearest integer
        ixs = [int(ix.item() + 0.5) for ix in ixs]
        return ixs

    def forward(self, xs: dict, **kwargs) -> tuple:
        """
        Perform a forward pass in the model
        :param xs: a dictionary of samples
        :param kwargs:
        :return: a two-tuple of:
                    - a tensor of indices (assumes daily time steps)
                    - a dictionary of additional info
        """
        raise NotImplementedError

    @classmethod
    def fit(cls,
            target_key: str,
            dataset: Dataset,
            # dataset_val: Dataset = None,
            model_name: str = None,
            num_epochs: int = 1,
            batch_size: int = 1,
            val_period: int = None,
            plot_period: int = None,
            scheduler_step_size: int = None,
            scheduler_decay: float = 0.5,
            clip_gradient: float = None,
            optimizer: str = None,
            optimizer_kwargs: dict = None,
            early_stopping: bool = True,
            early_stopping_patience: int = 1,
            early_stopping_min_delta: float = 0,
            model: 'BaseTorchModel' = None,
            model_kwargs: dict = None,
            device=torch.device('cpu'),
            seed: int = None,
            verbose: bool = True,
            ) -> tuple:
        """

        :param target_key:
        :param dataset:
        :param model_name:
        :param num_epochs:
        :param batch_size:
        :param val_period:
        :param plot_period:
        :param scheduler_step_size:
        :param scheduler_decay:
        :param clip_gradient:
        :param optimizer:
        :param optimizer_kwargs:
        :param early_stopping:
        :param early_stopping_patience:
        :param early_stopping_min_delta:
        :param model:
        :param model_kwargs:
        :param device:
        :param seed:
        :param verbose:
        :return:
        """
        # Validate input
        assert num_epochs > 0
        """
        Fill in missing input
        """
        if batch_size is None:
            batch_size = len(dataset)
        if val_period is None:
            val_period = np.inf
            dataset_trn, dataset_val = dataset, dataset
        else:
            dataset_trn, dataset_val = cls._split_dataset(dataset, seed=seed)
        if plot_period is None:
            plot_period = np.inf
        if optimizer is None:
            optimizer, optimizer_kwargs = cls._default_optimizer_w_kwargs()
        if optimizer_kwargs is None:
            optimizer_kwargs = dict()
        scheduler_step_size = scheduler_step_size or num_epochs

        model = (model or cls(**model_kwargs)).to(device)

        # Store model fit metadata in a dict
        fit_info = {
            'epochs': [],
        }

        optimizer = cls._get_optimizer(model, optimizer, optimizer_kwargs)

        scheduler = StepLR(optimizer,
                           step_size=scheduler_step_size,
                           gamma=scheduler_decay,
                           )

        stopping_criterion = EarlyStopper(patience=early_stopping_patience,
                                          min_delta=early_stopping_min_delta,
                                          )

        time_start = datetime.now()

        continue_loop = True
        for epoch in range(1, num_epochs + 1):
            if not continue_loop:
                break

            epoch_info = model._run_fit_epoch(target_key=target_key,
                                              model_name=model_name,
                                              dataset=dataset_trn,
                                              optimizer=optimizer,
                                              scheduler=scheduler,
                                              batch_size=batch_size,
                                              epoch_nr=epoch,
                                              num_epochs=num_epochs,
                                              clip_gradient=clip_gradient,
                                              device=device,
                                              verbose=verbose,
                                              )

            if dataset_val is not None and (epoch % val_period) == 0:
                val_info = model._run_eval(target_key=target_key,
                                           model_name=model_name,
                                           dataset=dataset_val,
                                           batch_size=batch_size,
                                           epoch_nr=epoch,
                                           device=device,
                                           verbose=verbose,
                                           )

                val_loss = val_info['loss']

                epoch_info['val'] = val_info

                if early_stopping and stopping_criterion.early_stop(val_loss):
                    continue_loop = False

            fit_info['epochs'].append(epoch_info)

            if (epoch % plot_period) == 0:
                model._savefigs_plot_losses(fit_info, model_name)

            if verbose and early_stopping and dataset_val is not None and (epoch % val_period) == 0:
                # desc = f'Epoch {epoch} / {num_epochs} | Time: {datetime.now() - time_start}'
                # desc += f' | Loss Min. {stopping_criterion.min_validation_loss:8.5f} | ES Count: {stopping_criterion.counter}/{stopping_criterion.patience} | ES Delta {stopping_criterion.min_delta}'
                # print(desc)
                min_loss_term = colored(f'{stopping_criterion.min_validation_loss:8.5f}', color='white', attrs=['bold'])
                desc = f'{model_name} ({cls.__name__})   {" " * 65} | Loss Min:  {min_loss_term} | ES Count: {stopping_criterion.counter}/{stopping_criterion.patience} | ES Delta {stopping_criterion.min_delta}'
                print(desc)

        # Model is assumed to be stored on CPU by default
        model.to(device=torch.device('cpu'))
        # Set the model to evaluation mode
        model.eval()

        time_end = datetime.now()

        # Complete info dict
        fit_info['time_passed'] = time_end - time_start
        fit_info['time_start'] = time_start
        fit_info['time_end'] = time_end

        # Return the fitted model, as well as info about the fitting procedure
        return model, fit_info

    @classmethod
    def _split_dataset(cls, dataset: Dataset, seed: int) -> Tuple[Dataset, Dataset]:
        yrs_trn, yrs_val = train_test_split(dataset.years, random_state=seed, shuffle=True)
        return dataset.select_years(years=yrs_trn), dataset.select_years(years=yrs_val)  # TODO -- move outside fit function

    def _run_fit_epoch(self,
                       target_key: str,
                       model_name: str,
                       dataset: Dataset,
                       optimizer: optim.Optimizer,
                       scheduler: optim.lr_scheduler.LRScheduler,
                       batch_size: int,
                       epoch_nr: int,
                       num_epochs: int,
                       clip_gradient: float = None,
                       device=torch.device('cpu'),
                       verbose: bool = True,
                       ) -> dict:
        """

        :param target_key:
        :param model_name:
        :param dataset:
        :param optimizer:
        :param scheduler:
        :param batch_size:
        :param epoch_nr:
        :param num_epochs:
        :param clip_gradient:
        :param device:
        :param verbose:
        :return:
        """
        time_start = datetime.now()

        dataset_wrapped = TorchDataset(dataset)

        dataloader = torch.utils.data.DataLoader(dataset_wrapped,
                                                 batch_size=batch_size,
                                                 collate_fn=dataset_wrapped.collate_fn,
                                                 shuffle=True,
                                                 )

        if verbose:
            train_iter = tqdm(dataloader,
                              total=len(dataloader),
                              )
        else:
            train_iter = dataloader

        losses = []

        self.to(device=device)
        self.train()

        for xs in train_iter:

            xs = TorchDataset.sample_to_device(xs, device)

            optimizer.zero_grad()

            loss, _ = self.loss(xs, target_key)

            loss.backward()

            if clip_gradient is not None:
                torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=clip_gradient)

            optimizer.step()

            losses.append(loss.item())
            loss_mean = sum(losses) / len(losses)

            lr = scheduler.get_last_lr()[0]

            if verbose:
                train_iter.set_description(
                    f'{model_name} ({self.__class__.__name__}) training epoch [{epoch_nr:6d}/{num_epochs}] | lr: {lr:.7f} | Batch Loss: {loss.item():8.5f} | Loss Mean: {loss_mean:8.5f}',
                )

        scheduler.step()

        time_end = datetime.now()

        return {
            'epoch': epoch_nr,
            'loss': sum(losses) / len(losses),
            'time_passed': time_end - time_start,
            'time_start': time_start,
            'time_end': time_end,
        }

    def _run_eval(self,
                  target_key: str,
                  model_name: str,
                  dataset: Dataset,
                  batch_size: int,
                  epoch_nr: int,
                  device=torch.device('cpu'),
                  verbose: bool = True,
                  ) -> dict:

        time_start = datetime.now()

        dataset_wrapped = TorchDataset(dataset)

        dataloader = torch.utils.data.DataLoader(dataset_wrapped,
                                                 batch_size=batch_size,
                                                 collate_fn=dataset_wrapped.collate_fn,
                                                 shuffle=False,
                                                 )

        if verbose:
            eval_iter = tqdm(dataloader,
                             total=len(dataloader),
                             )
        else:
            eval_iter = dataloader

        self.to(device=device)
        self.eval()

        losses = []
        ys_pred = []
        ys_true = []

        with torch.no_grad():

            for xs in eval_iter:
                xs = TorchDataset.sample_to_device(xs, device)

                loss, info = self.loss(xs, target_key)
                losses.append(loss.item())
                loss_mean = sum(losses) / len(losses)

                ys_pred.append(info['ys_pred'].detach().cpu().numpy())
                ys_true.append(info['ys_true'].detach().cpu().numpy())

                if verbose:
                    eval_iter.set_description(
                        f'{model_name} ({self.__class__.__name__}) validation {" " * 33} | Batch Loss: {loss.item():8.5f} | Loss Mean: {loss_mean:8.5f}',
                    )

        ys_pred = np.concatenate(ys_pred, axis=0)
        ys_true = np.concatenate(ys_true, axis=0)

        mae = mean_absolute_error(ys_true, ys_pred)
        mse = mean_squared_error(ys_true, ys_pred)
        r2 = r2_score(ys_true, ys_pred)
        if verbose:
            loss_term = colored(f'{sum(losses) / len(losses):8.5f}', color='green', attrs=['bold'])
            desc = f'{model_name} ({self.__class__.__name__})   {" " * 65} | Loss Mean: {loss_term} | MAE: {mae:.2f} | MSE: {mse:.2f} | r²: {r2:.2f}'
            print(desc)

        time_end = datetime.now()

        return {
            'epoch': epoch_nr,
            'loss': sum(losses) / len(losses),
            'time_passed': time_end - time_start,
            'time_start': time_start,
            'time_end': time_end,
            'mae': mae,
            'mse': mse,
            'r2': r2,
        }

    def loss(self, xs: dict, target_key: str, **kwargs: dict) -> tuple:

        ys_pred, info = self(xs)
        ys_true = xs[config.KEY_OBSERVATIONS_INDEX][target_key]

        loss = F.mse_loss(ys_pred, ys_true)

        return loss, {
            'forward_pass': info,
            'ys_pred': ys_pred,
            'ys_true': ys_true,
        }

    @classmethod
    def _default_optimizer_w_kwargs(cls) -> tuple:
        return 'adam', {
            'lr': 1e-3,
            'weight_decay': 1e-4,
        }

    @classmethod
    def _get_optimizer(cls, model: 'BaseTorchModel', optimizer_name: str, optimizer_kwargs: dict) -> optim.Optimizer:
        match optimizer_name:
            case 'adam':
                return torch.optim.Adam(model.parameters(), **optimizer_kwargs)
            case 'sgd':
                return torch.optim.SGD(model.parameters(), **optimizer_kwargs)
            case _:
                raise ModelException(f'Unknown optimizer: {optimizer_name}')

    def save(self, model_name: str, save_weights: bool = True, **kwargs: dict) -> None:

        if save_weights:
            self.save_weights(model_name)

        path = self._get_path(model_name)

        # Sub-folder for storing the entire model
        path_model = os.path.join(path, 'model')
        os.makedirs(path_model, exist_ok=True)

        torch.save(self, os.path.join(path_model, 'model.pt'))

    def save_weights(self, model_name: str, **kwargs: dict) -> None:
        """
        Save model weights/state dict
        :param model_name: model name
        """
        path = self._get_path(model_name)
        # Sub-folder for storing model weights
        path_weights = os.path.join(path, 'weights')
        os.makedirs(path_weights, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path_weights, 'state_dict.pth'))

    @classmethod
    def load(cls, model_name: str, **kwargs: dict) -> tuple:

        path = cls._get_path(model_name)

        # Sub-folder for storing the entire model
        path_model = os.path.join(path, 'model')

        model = torch.load(path_model)

        return model, {}

    @classmethod
    def load_weights(cls, model_name: str) -> dict:
        """
        Load model weights/state dict
        :param model_name: model name
        :return: the state dict
        """
        path = cls._get_path(model_name)
        return torch.load(os.path.join(path, 'weights', 'state_dict.pth'))

    @classmethod
    def _get_path(cls, model_name: str) -> str:  # TODO -- move to basemodel?
        return os.path.join(config.PATH_MODELS, cls.__name__, model_name)

    @classmethod
    def normalize_sample(cls, sample: dict, features_statistics: dict, in_place: bool = True) -> dict:

        norm_fs = dict()
        for key, f in sample[config.KEY_FEATURES].items():
            mean, std = features_statistics[key]
            norm_fs[key] = (f - mean) / std

        if in_place:
            sample[config.KEY_FEATURES] = norm_fs
            return sample
        else:
            sample = dict(sample)
            sample[config.KEY_FEATURES] = norm_fs
            return sample

    @classmethod
    def _savefigs_plot_losses(cls, info: dict, model_name: str, window_size: int = 100) -> None:
        assert window_size > 0

        x_train = []
        y_train = []

        x_eval = []
        y_eval = []

        for epoch, ei in enumerate(info['epochs']):
            epoch = epoch + 1

            x_train.append(epoch)
            y_train.append(ei['loss'])

            if 'val' in ei.keys():
                x_eval.append(epoch)
                y_eval.append(ei['val']['loss'])

        """
            Plot loss over entire training process
        """
        path = os.path.join(cls._get_path(model_name), 'fitting')
        os.makedirs(path, exist_ok=True)

        fig, ax = plt.subplots()

        ax.plot(x_train, y_train, c='b', label='Loss (Train)')
        ax.plot(x_eval, y_eval, c='r', label='Loss (Eval)')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

        ax.legend()

        plt.savefig(os.path.join(path, f'loss_progression.png'))

        plt.cla()
        plt.close()
        """
            Plot loss over previous window in training process
        """
        fig, ax = plt.subplots()

        x_window = x_train[-window_size:]
        y_window = y_train[-window_size:]

        ax.plot(x_window, y_window, c='b', label='Loss (Train)')
        ax.plot([x for x in x_eval if x >= x_window[0]],
                [y for i, y in enumerate(y_eval) if x_eval[i] >= x_window[0]],
                c='r', label='Loss (Eval)')
        # ax.plot(x_eval, y_eval, c='r', label='Loss (Eval)')

        if len(y_eval) != 0:
            plt.axhline(min(y_eval), color='grey', linestyle='--')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

        ax.set_xlim(max(0, x_train[-1] - window_size), x_train[-1])

        ax.legend()

        plt.savefig(os.path.join(path, f'loss_progression_window.png'))

        plt.cla()
        plt.close()


class NullTorchModel(BaseTorchModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._param = nn.Parameter(torch.zeros(1))

    def forward(self, xs: dict, **kwargs) -> tuple:

        fs = torch.cat([f for f in xs[config.KEY_FEATURES].values()], dim=1)
        ix = (fs * self._param).sum(dim=1)
        ix = torch.clamp(ix, min=0, max=365)
        return ix, {}


