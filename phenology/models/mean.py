
from collections import defaultdict

import numpy as np

import phenology.config as config_data
from phenology.models.base import BaseModel

from phenology.dataset.dataset import Dataset


class MeanModel(BaseModel):

    def __init__(self):
        super().__init__()
        self._mean = None

    def predict(self, sample: dict, **kwargs: dict) -> tuple:
        assert self._mean is not None, "Mean has not been fit"

        date = sample['season_start'] + np.timedelta64(int(self._mean), 'D')

        return date, {
            'ix': int(self._mean),
        }

    @classmethod
    def fit(cls,
            target_key: str,
            dataset: Dataset,
            model_name: str = None,
            model: 'MeanModel' = None,
            model_kwargs: dict = None,
            **kwargs: dict,
            ) -> tuple:

        if model is None:
            model = cls._init_with_required_kwargs(
                kwargs=model_kwargs if model_kwargs is not None else {},
                required=[],
            )

        ys = []
        for x in dataset.iter_items():
            y = x[config_data.KEY_OBSERVATIONS_INDEX][target_key]
            ys.append(y)
        model._mean = sum(ys) / len(ys)
        return model, {}


class YearMeanModel(BaseModel):

    def __init__(self,):
        super().__init__()
        self._means = None

    def predict(self, sample: dict, **kwargs: dict) -> tuple:
        assert self._means is not None, "Mean has not been fit"

        year = sample[config_data.KEY_YEAR]

        mean_ix = self._means[year]

        date = sample['season_start'] + np.timedelta64(int(mean_ix), 'D')

        return date, {
            'ix': int(mean_ix),
        }

    @classmethod
    def fit(cls,
            target_key: str,
            dataset: Dataset,
            model_name: str = None,
            model: 'YearMeanModel' = None,
            model_kwargs: dict = None,
            **kwargs: dict,
            ) -> tuple:

        if model is None:
            model = cls._init_with_required_kwargs(
                kwargs=model_kwargs if model_kwargs is not None else {},
                required=[],
            )

        ys_year = defaultdict(list)  # Observations per year

        for x in dataset.iter_items():
            y = x[config_data.KEY_OBSERVATIONS_INDEX][target_key]
            year = x[config_data.KEY_YEAR]

            ys_year[year].append(y)

        means = {
            year: sum(ys_year[year]) / len(ys_year[year]) for year in ys_year.keys()
        }

        model._means = means
        return model, {}

