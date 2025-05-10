import numpy as np
from sklearn.linear_model import LinearRegression

from phenology import config
from phenology.dataset.dataset import Dataset
from phenology.models.base import BaseModel


class LinearTrendModel(BaseModel):

    def __init__(self,
                 linear_kwargs: dict = None,
                 ):
        super().__init__()
        if linear_kwargs is None:
            linear_kwargs = dict()

        self._model = LinearRegression(**linear_kwargs)

    def predict(self, sample: dict, **kwargs: dict) -> tuple:

        year = sample[config.KEY_YEAR]

        out = self._model.predict(np.array([[year]]))

        # Remove data index and round to nearest integer
        ix = int(out.reshape(-1) + 0.5)

        # convert the index to a date object for evaluation
        fmt = 'D'  # daily time steps
        date = sample['season_start'] + np.timedelta64(int(ix), fmt)

        return date, {
            'ix': ix,
        }

    @classmethod
    def fit(cls,
            target_key: str,
            dataset: Dataset,
            model_name: str = None,
            model: 'BaseModel' = None,
            model_kwargs: dict = None,
            **kwargs: dict,
            ) -> tuple:

        if model is None:
            model = cls._init_with_required_kwargs(
                kwargs=model_kwargs if model_kwargs is not None else {},
                required=[],
            )

        if len(dataset) == 0:
            return model, {

            }

        x = []  # Year
        y = []  # Index

        for sample in dataset.iter_items():
            year = sample[config.KEY_YEAR]
            obs = sample[config.KEY_OBSERVATIONS_INDEX][target_key]

            x.append(year)
            y.append(obs)

        model._model.fit(np.reshape(np.array(x), (-1, 1)), np.array(y))

        return model, {

        }