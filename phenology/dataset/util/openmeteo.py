import os.path
from functools import lru_cache

import numpy as np
import pandas as pd
from tqdm import tqdm

from phenology import config

from phenology.data.openmeteo.download import OpenMeteoStores
from phenology.dataset.util.calendar import Calendar


class OpenMeteoDataset:

    """



    """

    def __init__(self,
                 calendar: Calendar,
                 use_cache: bool = False,
                 cache_index: list = None,
                 step: str = None,
                 data_keys: list = None,
                 save: bool = False,
                 debug_mode: bool = False,
                 ):

        if use_cache or cache_index is not None or save:
            print('warning -- dataset caching is bugged and disabled atm!')

        self._debug_mode = debug_mode
        self._calendar = calendar
        self._use_cache = use_cache

        if not self._debug_mode:
            step_key_pairs = list(zip([step] * len(data_keys), data_keys))

            self._store = OpenMeteoStores(step_key_pairs)
            self._df = None  # Not used
        else:
            self._store = None
            self._df = None
        #
        # use_cache = False  # TODO -- this is broken atm!
        # save = False
        # cache_index = None
        #
        # self._calendar = calendar
        # self._use_cache = use_cache
        # self._store = None
        # self._debug_mode = debug_mode
        #
        # if not self._debug_mode:
        #     step_key_pairs = list(zip([step] * len(data_keys), data_keys))
        #
        #     if use_cache:
        #         if self._save_exists():
        #             self._df = self._load_save()
        #         else:  # TODO -- special case where either len(index) or len(keys) is 0!
        #             self._store = OpenMeteoStores(step_key_pairs)
        #             self._df = self._create_df(step, data_keys, cache_index)
        #             self._df.sort_index(inplace=True)
        #             self.close()
        #     else:
        #         self._store = OpenMeteoStores(step_key_pairs)
        #         self._df = None  # Not used
        #
        #     if save:
        #         self._save(step, data_keys, cache_index)
        # else:
        #     self._store = None
        #     self._df = None

    def _create_df(self, step: str, data_keys: list, cache_index: list, verbose: bool = True) -> pd.DataFrame:
        assert step is not None, 'step is required.'
        assert data_keys is not None, 'data_keys is required.'
        assert cache_index is not None, 'index is required to create cache'

        data = []

        if verbose:
            index_iter = tqdm(cache_index,
                              total=len(cache_index),
                              desc='Creating cache DataFrame. Fetching data from store',
                              )
        else:
            index_iter = cache_index

        for i in index_iter:
            src, loc_id, year, species_code, subgroup_code = i
            for key in data_keys:

                data.append({
                    'step': step,
                    'key': key,
                    config.KEY_DATA_SOURCE: src,
                    config.KEY_LOC_ID: loc_id,
                    config.KEY_YEAR: year,
                    config.KEY_SPECIES_CODE: species_code,
                    config.KEY_SUBGROUP_CODE: subgroup_code,
                    'data': self._get_from_store(step, key, i)
                })

        df = pd.DataFrame(data)
        df.set_index([
                              'step',
                              'key',
                              config.KEY_DATA_SOURCE,
                              config.KEY_LOC_ID,
                              config.KEY_YEAR,
                              config.KEY_SPECIES_CODE,
                              config.KEY_SUBGROUP_CODE,
                          ], inplace=True)

        df.sort_index(inplace=True)

        return df

    def _save(self, step: str, data_keys: list, cache_index: list):
        if self._df is not None:
            df = self._df
        else:
            df = self._create_df(step, data_keys, cache_index)

            df['data'] = df['data'].apply(lambda x: list(x))  # Cast np array to list

        df.to_csv(self._save_path)

    def _load_save(self) -> pd.DataFrame:

        df = pd.read_csv(self._save_path,
                         index_col=[
                             'step',
                             'key',
                             config.KEY_DATA_SOURCE,
                             config.KEY_LOC_ID,
                             config.KEY_YEAR,
                             config.KEY_SPECIES_CODE,
                             config.KEY_SUBGROUP_CODE,
                         ],
                         )

        df.sort_index(inplace=True)

        df['data'] = df['data'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))  # parse str array

        return df

    def _save_exists(self):
        return os.path.exists(self._save_path)

    @property
    def _save_path(self) -> str:
        path = config.PATH_DATASETS
        name = 'openmeteo.csv'
        path = os.path.join(path, name)
        return path

    @lru_cache(maxsize=None)  # Cache least-recently-used/requested data
    def get_data(self, step: str, key: str, index: tuple) -> np.ndarray:

        if self._debug_mode:
            return self._get_debug_data(step, key, index)

        if self._use_cache:
            return self._get_from_df(step, key, index)
        else:
            return self._get_from_store(step, key, index)

    def _get_debug_data(self, step: str, data_key: str, index: tuple) -> np.ndarray:

        src, loc_id, year, species_code, subgroup_code = index

        season = self._get_season_info(src, loc_id, year, species_code, subgroup_code)

        step_fmt = 'h' if step == 'hourly' else 'D'

        season_start = season['season_start']
        season_end = season['season_end'] - np.timedelta64(1, step_fmt)

        return np.zeros((season_end - season_start) // np.timedelta64(1, step_fmt))

    def _get_from_store(self, step: str, data_key: str, index: tuple):

        src, loc_id, year, species_code, subgroup_code = index

        season = self._get_season_info(src, loc_id, year, species_code, subgroup_code)

        step_fmt = 'h' if step == 'hourly' else 'D'

        season_start = season['season_start']
        season_end = season['season_end'] - np.timedelta64(1, step_fmt)

        year_min = season_start.astype('datetime64[Y]').astype(int) + 1970
        year_max = season_end.astype('datetime64[Y]').astype(int) + 1970

        season_start = pd.Timestamp(season_start, tz='UTC')
        season_end = pd.Timestamp(season_end, tz='UTC')

        df_x = pd.concat([self._store[step, data_key, src, loc_id, year_].bfill().ffill() for year_ in range(year_min, year_max + 1)],  # TODO -- the bfill should be postprocessing
                         axis=0,
                         )
        df_x.set_index('date', inplace=True)

        df_x = df_x[season_start:season_end]

        return df_x[data_key].values

    def _get_from_df(self, step: str, data_key: str, index: tuple) -> np.ndarray:
        src, loc_id, year, species_code, subgroup_code = index
        entry = self._df.loc[step, data_key, src, loc_id, year, species_code, subgroup_code]['data']
        return entry

    # @lru_cache(maxsize=None)  # Cache least-recently-used/requested data
    def _get_season_info(self, src, loc_id, year, species_code, subgroup_code):
        season = self._calendar.get_season_info(year=year,
                                                species_code=species_code,
                                                subgroup_code=subgroup_code,
                                                src=src,
                                                loc_id=loc_id,
                                                )
        return season

    def __getitem__(self, i: tuple) -> np.ndarray:
        step, key, index = i
        return self.get_data(step, key, index)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._store is not None:
            self._store.close()
        self._store = None
