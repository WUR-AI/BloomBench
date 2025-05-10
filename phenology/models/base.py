import os.path
from abc import ABC, abstractmethod

import pickle

import numpy as np

from phenology import config
from phenology.dataset.dataset import Dataset
from phenology.models import ModelException


class BaseModel(ABC):

    @abstractmethod
    def predict(self, sample: dict, **kwargs: dict) -> tuple:
        """
        Given a data sample, predict the moment of phenological transition

        :param sample: a dict corresponding to a single data entry
        :param kwargs: possible additional keyword arguments
        :return: a two-tuple, containing:
            - the predicted moment as np.datetime64 object
            - a dict containing additional info
        """
        raise NotImplementedError

    def predict_all(self, samples: list, **kwargs: dict) -> list:
        return [self.predict(sample, **kwargs) for sample in samples]

    @classmethod
    @abstractmethod
    def fit(cls,
            target_key: str,
            dataset: Dataset,
            model_name: str = None,
            model: 'BaseModel' = None,
            model_kwargs: dict = None,
            **kwargs: dict,
            ) -> tuple:
        """
        Fit a model to the given dataset
        :param target_key: the key of the target variable
        :param dataset: the dataset to fit to
        :param model_name: optional name of the model (class name by default)
        :param model: optional existing model instance to fit instead of creating a new one by default
        :param model_kwargs: keyword arguments to pass to the model initialization. None are passed if left empty
        :param kwargs: optional additional keyword arguments for the fitting procedure
        :return: a tuple, containing:
            - the fitted model
            - a dict containing additional info
        """
        raise NotImplementedError

    def save(self, model_name: str, **kwargs: dict) -> None:
        fn = f'{model_name}.pickle'
        # Define and optionally create the folder where the model should be stored
        folder_path = os.path.join(config.PATH_MODELS, model_name)
        os.makedirs(folder_path, exist_ok=True)
        # Write the file to the folder
        with open(os.path.join(folder_path, fn), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_name: str, **kwargs: dict) -> tuple:
        fn = f'{model_name}.pickle'
        path = os.path.join(config.PATH_MODELS, model_name, fn)
        with open(path, 'rb') as f:
            return pickle.load(f), {}

    @classmethod
    def _init_with_required_kwargs(cls, kwargs: dict, required: list) -> 'BaseModel':
        """
        Convenience method to initialize a model with required kwargs. Checks whether required kwargs are present.
        :param kwargs: The kwargs to initialize the model with
        :param required: A list of required kwargs
        :return: The initialized model
        """
        if any([kwarg not in kwargs for kwarg in required]):
            raise ModelException(f'Required keyword arguments not present for class {cls.__name__}.'
                                 f' Required: {required}, Obtained: {list(kwargs.keys())}.')
        return cls(**kwargs)


class NullModel(BaseModel):
    """
    For debugging purposes
    """

    def predict(self, sample: dict, **kwargs: dict) -> tuple:
        return np.datetime64(0, 'Y'), {}

    @classmethod
    def fit(cls,
            target_key: str,
            dataset: Dataset,
            model_name: str = None,
            model: 'NullModel' = None,
            model_kwargs: dict = None,
            **kwargs: dict,
            ) -> tuple:
        return NullModel(), {}
