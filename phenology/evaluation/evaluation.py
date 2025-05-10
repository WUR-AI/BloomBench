import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

import json

from tqdm import tqdm

from phenology import config
from phenology.dataset.dataset import Dataset
from phenology.models.base import BaseModel


class EvaluationResult:

    def __init__(self,
                 run_name: str,
                 model_name: str,
                 df_dates_trn: pd.DataFrame,
                 df_dates_tst: pd.DataFrame,
                 metadata: dict = None,
                 ):
        if metadata is None:
            metadata = dict()

        self._df_dates_trn = df_dates_trn
        self._df_dates_tst = df_dates_tst

        self._info = {
            'run_name': run_name,
            'model_name': model_name,
            'timestamp': pd.Timestamp('now').strftime('%Y-%m-%d %X'),
            **metadata,
        }

    @property
    def run_name(self) -> str:
        return self._info['run_name']

    @property
    def model_name(self) -> str:
        return self._info['model_name']

    @property
    def timestamp(self) -> str:
        return self._info['timestamp']

    @property
    def df_dates_trn(self) -> pd.DataFrame:
        return self._df_dates_trn

    @property
    def df_dates_tst(self) -> pd.DataFrame:
        return self._df_dates_tst

    def save(self):
        path = os.path.join(config.PATH_OUTPUT_EVAL, self.run_name)
        os.makedirs(path, exist_ok=True)
        self._df_dates_trn.to_csv(os.path.join(path, 'dates_train.csv'))
        self._df_dates_tst.to_csv(os.path.join(path, 'dates_test.csv'))

        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(self._info, f)

    @staticmethod
    def load(run_name: str) -> 'EvaluationResult':

        path = os.path.join(config.PATH_OUTPUT_EVAL, run_name)
        date_columns = ['date_true', 'date_pred', 'season_start', 'season_end']
        df_trn = pd.read_csv(os.path.join(path, 'dates_train.csv'),
                             index_col=['src', 'loc_id', 'year', 'species_code', 'subgroup_code'],
                             parse_dates=date_columns,
                             )
        df_tst = pd.read_csv(os.path.join(path, 'dates_test.csv'),
                             index_col=['src', 'loc_id', 'year', 'species_code', 'subgroup_code'],
                             parse_dates=date_columns,
                             )

        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            info = json.load(f)

        model_name = info['model_name']

        result = EvaluationResult(
            run_name=run_name,
            model_name=model_name,
            df_dates_trn=df_trn,
            df_dates_tst=df_tst,
        )
        result._info = info
        return result

    def compute_metrics(self,
                        metrics: dict = None,
                        ) -> dict:

        column_name = self.model_name or 'value'
        metrics = metrics or self._default_metrics()
        rows_trn = []
        rows_tst = []

        # TODO -- this would break if dates are around end of dec/start of jan
        y_pred_trn = self._df_dates_trn['date_pred'].dt.dayofyear
        y_true_trn = self._df_dates_trn['date_true'].dt.dayofyear

        y_pred_tst = self._df_dates_tst['date_pred'].dt.dayofyear
        y_true_tst = self._df_dates_tst['date_true'].dt.dayofyear

        for metric, func in metrics.items():
            val_trn = func(y_true_trn, y_pred_trn)
            val_tst = func(y_true_tst, y_pred_tst)

            rows_trn.append({
                'metric': metric,
                column_name: val_trn,
            })
            rows_tst.append({
                'metric': metric,
                column_name: val_tst,
            })

        df_trn = pd.DataFrame(rows_trn)
        df_tst = pd.DataFrame(rows_tst)

        df_trn.set_index('metric', inplace=True)
        df_tst.set_index('metric', inplace=True)

        return {
            'train': df_trn,
            'test': df_tst,
        }

    @staticmethod
    def _default_metrics() -> dict:
        return {
            'mse': lambda _doy_true, _doy_pred: mean_squared_error(_doy_true, _doy_pred),
            'rmse': lambda _doy_true, _doy_pred: root_mean_squared_error(_doy_true, _doy_pred),
            'mae': lambda _doy_true, _doy_pred: mean_absolute_error(_doy_true, _doy_pred),
            'r²': lambda _doy_true, _doy_pred: r2_score(_doy_true, _doy_pred),
            'kendall_tau': lambda _doy_true, _doy_pred: kendalltau(_doy_true, _doy_pred).statistic,
            'mean': lambda _doy_true, _doy_pred: np.mean(_doy_pred),
            'std': lambda _doy_true, _doy_pred: np.std(_doy_pred),
            'count': lambda _doy_true, _doy_pred: len(_doy_true),
        }

    def savefig_scatter(self,
                        path,
                        fn: str = None,
                        as_doy: bool = True,
                        ) -> None:
        fn = fn or 'scatter.png'
        df_trn = self.df_dates_trn
        df_tst = self.df_dates_tst

        dfs_metrics = self.compute_metrics()

        mse_trn = dfs_metrics['train'].loc['mse'].values[0]
        mse_tst = dfs_metrics['test'].loc['mse'].values[0]

        mae_trn = dfs_metrics['train'].loc['mae'].values[0]
        mae_tst = dfs_metrics['test'].loc['mae'].values[0]

        r2_trn = dfs_metrics['train'].loc['r²'].values[0]
        r2_tst = dfs_metrics['test'].loc['r²'].values[0]

        if as_doy:
            y_true_trn = df_trn['date_true'].dt.dayofyear
            y_pred_trn = df_trn['date_pred'].dt.dayofyear

            y_true_tst = df_tst['date_true'].dt.dayofyear
            y_pred_tst = df_tst['date_pred'].dt.dayofyear

        else:

            y_true_trn = (df_trn['date_true'] - df_trn['season_start']) // np.timedelta64(1, 'D')
            y_pred_trn = (df_trn['date_pred'] - df_trn['season_start']) // np.timedelta64(1, 'D')

            y_true_tst = (df_tst['date_true'] - df_tst['season_start']) // np.timedelta64(1, 'D')
            y_pred_tst = (df_tst['date_pred'] - df_tst['season_start']) // np.timedelta64(1, 'D')

        # Create a list of all predicted and true values
        values = np.concatenate([
            y_true_trn,
            y_pred_trn,
            y_true_tst,
            y_pred_tst,
        ])
        # Use the list to define the range of the axes
        buffer = 5  # days
        val_max = max(values) + buffer
        val_min = min(values) - buffer
        val_range = np.arange(val_min, val_max)

        fig, axs = plt.subplots(nrows=2, figsize=(6, 10), sharey=True, sharex=True)
        ax_trn, ax_tst = axs

        alpha = 0.1
        dot_size = 2
        dot_color = 'black'

        ax_trn.scatter(y_true_trn,
                       y_pred_trn,
                       s=dot_size,
                       alpha=alpha,
                       c=dot_color,
                       label=f'MSE: {mse_trn:.2f}, MAE: {mae_trn:.2f}, $r^2$: {r2_trn:.2f}',
                       )
        ax_trn.plot(val_range, val_range, '--', c='lightgray')

        ax_tst.scatter(y_true_tst,
                       y_pred_tst,
                       s=dot_size,
                       alpha=alpha,
                       c=dot_color,
                       label=f'MSE: {mse_tst:.2f}, MAE: {mae_tst:.2f}, $r^2$: {r2_tst:.2f}',
                       )
        ax_tst.plot(val_range, val_range, '--', c='lightgray')

        ax_trn.set_title('Train')
        ax_tst.set_title('Test')

        if as_doy:
            for ax in axs:
                ax.set_xlabel('True DoY')
                ax.set_ylabel('Predicted DoY')
        else:
            for ax in axs:
                ax.set_xlabel('True Index')
                ax.set_ylabel('Predicted Index')

        for ax in axs:
            ax.set_xlim(val_min, val_max)
            ax.set_ylim(val_min, val_max)

        ax_trn.legend()
        ax_tst.legend()

        # fig.suptitle(self.run_name)

        os.makedirs(os.path.join(path, self.run_name), exist_ok=True)
        plt.savefig(os.path.join(path, self.run_name, fn))
        plt.close()
        plt.cla()


def evaluate(model: BaseModel,
             dataset_train: Dataset,
             dataset_test: Dataset,
             target,
             run_name: str,
             model_name: str = None,
             ) -> EvaluationResult:
    model_name = model_name or model.__class__.__name__

    dfs = _run_eval(
        model=model,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        target=target,
        model_name=model_name,
    )

    df_trn = dfs['train']
    df_tst = dfs['test']

    return EvaluationResult(
        run_name=run_name,
        model_name=model_name,
        df_dates_trn=df_trn,
        df_dates_tst=df_tst,
    )


def _run_eval(model: BaseModel,
              dataset_train: Dataset,
              dataset_test: Dataset,
              target,
              model_name: str,
              set_index: bool = True,
              ) -> dict:

    df_trn = _eval_on_dataset(
        model=model,
        dataset=dataset_train,
        target=target,
        model_name=model_name,
        set_index=set_index,
        desc=f"Evaluating {model_name} ({model.__class__.__name__}) on the train set",
    )

    df_tst = _eval_on_dataset(
        model=model,
        dataset=dataset_test,
        target=target,
        model_name=model_name,
        set_index=set_index,
        desc=f"Evaluating {model_name} ({model.__class__.__name__}) on the test set",
    )

    return {
        'train': df_trn,
        'test': df_tst,
    }


def _eval_on_dataset(model: BaseModel,
                     dataset: Dataset,
                     target,
                     model_name: str,
                     set_index: bool = True,
                     desc: str = None,
                     ) -> pd.DataFrame:

    desc = desc or f"Evaluating {model_name} ({model.__class__.__name__})"

    rows = []

    for sample in tqdm(dataset.iter_items(),
                       desc=desc,
                       total=len(dataset),
                       ):

        # Some data points have features with NaN that cannot be imputed. Ignore these for evaluation
        # TODO -- this check should be in the dataset definition
        nan_found = False
        for f in sample[config.KEY_FEATURES].values():
            nan_found = nan_found or np.isnan(f).any()
            if nan_found:
                break
        if nan_found:
            continue

        date_true = sample[config.KEY_OBSERVATIONS][target]
        date_pred, _ = model.predict(sample)

        rows.append({
            config.KEY_DATA_SOURCE: sample[config.KEY_DATA_SOURCE],
            config.KEY_LOC_ID: sample[config.KEY_LOC_ID],
            config.KEY_YEAR: sample[config.KEY_YEAR],
            config.KEY_SPECIES_CODE: sample[config.KEY_SPECIES_CODE],
            config.KEY_SUBGROUP_CODE: sample[config.KEY_SUBGROUP_CODE],
            'date_true': date_true,
            'date_pred': date_pred,
            'season_start': sample['season_start'],
            'season_end': sample['season_end'],
        })

    df = pd.DataFrame(rows)

    if set_index and len(df) > 0:
        df.set_index(
            [config.KEY_DATA_SOURCE,
             config.KEY_LOC_ID,
             config.KEY_YEAR,
             config.KEY_SPECIES_CODE,
             config.KEY_SUBGROUP_CODE,
             ], inplace=True,
        )

    return df

