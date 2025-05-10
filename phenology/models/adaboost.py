import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

import phenology.config as config
from phenology.dataset.dataset import Dataset
from phenology.models.base import BaseModel


class AdaBoostModel(BaseModel):

    def __init__(self,
                 adaboost_kwargs: dict = None,
                 ):
        if adaboost_kwargs is None:
            adaboost_kwargs = dict()

        self._model = AdaBoostRegressor(**adaboost_kwargs)

    def predict(self, sample: dict, **kwargs: dict) -> tuple:

        # Convert the sample dict to a vector
        fs = self._sample_to_vector(sample)

        # predict is intended to be called on multiple datapoints, but a single point is given
        # i.e. it expects a matrix of size N x M where N is nr of datapoints and M the nr of features
        fs = fs.reshape(1, -1)

        out = self._model.predict(fs)

        # Remove data index and round to nearest integer
        ix = int(out.reshape(-1) + 0.5)

        # convert the index to a date object for evaluation
        fmt = 'D'  # daily time steps
        date = sample['season_start'] + np.timedelta64(int(ix), fmt)

        return date, {
            'ix': ix,
        }

    def _sample_to_vector(self, sample: dict) -> np.ndarray:

        # get feature data
        #   temperature
        ts = sample[config.KEY_FEATURES]['temperature_2m_mean']
        #   photoperiod
        ps = sample[config.KEY_FEATURES]['daylight_duration'] / 3600  # Convert from seconds to hours

        fs = np.concatenate(
            [
                ts.reshape(-1, 1),
                ps.reshape(-1, 1),
            ],
            axis=1,
        )

        return fs

    @classmethod
    def fit(cls,
            target_key: str,
            dataset: Dataset,
            model_name: str = None,
            model: 'AdaBoostRegressor' = None,
            model_kwargs: dict = None,
            hf: bool = True,
            hf_n_iter: int = 10,
            hf_cv: int = 5,
            hf_random_state: int = 0,
            **kwargs: dict,
            ) -> tuple:
        """

        :param target_key:
        :param dataset:
        :param model_name:
        :param model:
        :param model_kwargs:
        :param hf: boolean indicating whether hyperparameters are fitted or not
        :param hf_n_iter: nr of iterations of random search over hyperparameters
        :param hf_cv: nr of folds used in random search over hyperparameters
        :param hf_random_state: random seed for hyperparameter optimization
        :param kwargs: additional keyword arguments
        :return: a two-tuple of (fitted model, info dict)
        """

        if model is None:
            model = cls._init_with_required_kwargs(
                kwargs=model_kwargs if model_kwargs is not None else {},
                required=[],
            )

        if len(dataset) == 0:
            return model, {

            }

        # Construct a matrix representation of the data
        #   1. Transform all data points to a vector representation
        #   2. Concatenate all vectors to form a matrix
        m = []  # List of vectors
        t = []  # List of corresponding targets
        for x in dataset.iter_items():
            v = model._sample_to_vector(x)
            if np.isnan(v).any():
                continue
            m.append(v)
            t.append(x[config.KEY_OBSERVATIONS_INDEX][target_key])
        m = np.concatenate([v.reshape(1, -1) for v in m], axis=0)
        t = np.array(t)

        if hf:

            distributions = {
                'n_estimators': randint(low=1, high=200 + 1),
            }

            clf = RandomizedSearchCV(model._model,
                                     distributions,
                                     random_state=hf_random_state,
                                     n_iter=hf_n_iter,
                                     cv=hf_cv,
                                     n_jobs=min(hf_n_iter, 10),
                                     verbose=4,
                                     )

            hf = clf.fit(m, t)

            print(hf.best_params_)

            model._model = AdaBoostRegressor(**hf.best_params_)

        model._model.fit(m, t)

        return model, {

        }
