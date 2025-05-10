from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
from tqdm import tqdm

from phenology import config
from phenology.data.openmeteo.download import OpenMeteoEntry, get_openmeteo_data
from phenology.dataset.base import BaseDataset
from phenology.dataset.util.calendar import Calendar
from phenology.dataset.util.openmeteo import OpenMeteoDataset


class Dataset(BaseDataset):

    DEFAULT_DATA_KEYS = ('temperature_2m_mean', 'daylight_duration')
    DEFAULT_STEP = 'daily'

    def __init__(self,
                 df_y: pd.DataFrame,
                 df_y_loc: pd.DataFrame,
                 data_openmeteo: OpenMeteoDataset = None,
                 calendar: Calendar = None,
                 mode_download_meteo: str = None,
                 data_keys: tuple = None,
                 step: str = None,
                 match_observation_ixs_to_step: bool = False,
                 debug_mode: bool = False,
                 ):
        """
        Create a new Dataset object. Preferred method is through Dataset.load
        :param mode_download_meteo:
        :param debug_mode:
        """
        assert step in (None, 'hourly', 'daily')
        super().__init__(df_y, df_y_loc)
        self._debug_mode = debug_mode
        self._match_obs_ixs_to_step = match_observation_ixs_to_step

        if data_keys is None:
            self._data_keys = self.DEFAULT_DATA_KEYS
        else:
            self._data_keys = data_keys

        if step is None:
            self._step = self.DEFAULT_STEP
        else:
            self._step = step

        self._calendar = calendar or Calendar.from_config()

        if data_openmeteo is None:
            self._obtain_openmeteo_data(mode_download_meteo)  # TODO -- move to preprocessing
            self._data_om = OpenMeteoDataset(
                calendar=self._calendar,
                # use_cache=load_cache_meteo,
                use_cache=False,
                # save=save_meteo_data,
                save=False,  # TODO -- caching disabled -- contains bugs
                # cache_index=self._index,
                data_keys=list(self._data_keys),
                step=self._step,
                debug_mode=debug_mode,
            )
        else:
            self._data_om = data_openmeteo

        # if (data_openmeteo is None and not self._debug_mode) or (data_openmeteo is None and len(data_keys) == 0):
        #     self._obtain_openmeteo_data(mode_download_meteo)
        #     self._store =
        # else:
        #     self._store = data_openmeteo

    def _obtain_openmeteo_data(self, download_mode: str):  # TODO -- move to preprocessing

        entries = set()

        for data_key in self._data_keys:

            for index in self.iter_index():
                src, loc_id, year, species_code, subgroup_code = index

                season = self._calendar.get_season_info(year=year,
                                                        species_code=species_code,
                                                        subgroup_code=subgroup_code,
                                                        src=src,
                                                        loc_id=loc_id,
                                                        )

                season_start = season['season_start']
                season_end = season['season_end']

                coords = self.get_location_coordinates(i=(src, loc_id))
                # cc, _ = self.get_location_country(i=(src, loc_id))

                year_min = season_start.astype('datetime64[Y]').astype(int) + 1970
                year_max = season_end.astype('datetime64[Y]').astype(int) + 1970

                for year_ in range(year_min, year_max + 1):

                    entries.add(
                        OpenMeteoEntry(
                            step=self._step,
                            data_key=data_key,
                            src_key=src,
                            loc_id=loc_id,
                            year=year_,
                            loc_name=self.get_location_name(i=(src, loc_id)),
                            lat=coords['lat'],
                            lon=coords['lon'],
                            # country_code=cc,
                        )
                    )

        result = get_openmeteo_data(entries=entries,
                                    verbose=True,
                                    download_mode=download_mode,
                                    )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._data_om is not None:
            self._data_om.close()

    def __getitem__(self, index) -> dict:
        # Obtain the observations as defined by the super class
        item = super(Dataset, self).__getitem__(index)

        # Obtain info from the data sample index
        src = item[config.KEY_DATA_SOURCE]
        loc_id = item[config.KEY_LOC_ID]
        year = item[config.KEY_YEAR]
        species_code = item[config.KEY_SPECIES_CODE]
        subgroup_code = item[config.KEY_SUBGROUP_CODE]

        index = src, loc_id, year, species_code, subgroup_code

        # Obtain the observations
        observations = item[config.KEY_OBSERVATIONS]

        # Obtain season info
        season = self._get_season_info(src, loc_id, year, species_code, subgroup_code)

        # Compute the start and end moment of the season
        # Start and end moment correspond to the first and last element in the time series features
        # (i.e. season end is included)
        step_fmt = 'h' if self._step == 'hourly' else 'D'
        # Get season start
        season_start = season['season_start']
        # Get season end (calendar entry marks the day at which the season has ended)
        # Subtract one time unit to make the datetime object match the last entry in the time series data
        season_end = season['season_end'] - np.timedelta64(1, step_fmt)

        # Provide observations as indices within the season, next to datetime objects
        # Depending on the configuration of this object, these indices are based on nr of days or the same time step as
        # the feature data
        step_fmt_obs = step_fmt if self._match_obs_ixs_to_step else 'D'
        obs_ixs = {
            key: (o - season_start) // np.timedelta64(1, step_fmt_obs) for key, o in observations.items()
        }

        # Return all relevant data
        return {
            **item,
            'season_start': season_start,
            'season_end': season_end,
            config.KEY_OBSERVATIONS_INDEX: obs_ixs,
            config.KEY_FEATURES: {
                key: self._get_feature(self._step, key, index) for key in self._data_keys
            },
        }

    # @lru_cache(maxsize=None)  # Cache least-recently-used/requested data
    def _get_feature(self, step: str, data_key: str, index: tuple) -> np.ndarray:
        return self._data_om.get_data(step, data_key, index)
        #
        # src, loc_id, year, species_code, subgroup_code = index
        #
        # season = self._get_season_info(src, loc_id, year, species_code, subgroup_code)
        #
        # step_fmt = 'h' if self._step == 'hourly' else 'D'
        #
        # season_start = season['season_start']
        # season_end = season['season_end'] - np.timedelta64(1, step_fmt)
        #
        # if self._debug_mode:
        #     return np.zeros((season_end - season_start) // np.timedelta64(1, step_fmt))
        #
        # year_min = season_start.astype('datetime64[Y]').astype(int) + 1970
        # year_max = season_end.astype('datetime64[Y]').astype(int) + 1970
        #
        # season_start = pd.Timestamp(season_start, tz='UTC')
        # season_end = pd.Timestamp(season_end, tz='UTC')
        #
        # df_x = pd.concat([self._store[step, data_key, src, loc_id, year_] for year_ in range(year_min, year_max + 1)],
        #                  axis=0,
        #                  )
        # df_x.set_index('date', inplace=True)
        #
        # df_x = df_x[season_start:season_end]
        #
        # return df_x[data_key].values

    # @lru_cache(maxsize=None)  # Cache least-recently-used/requested data
    def _get_season_info(self, src, loc_id, year, species_code, subgroup_code):
        season = self._calendar.get_season_info(year=year,
                                                species_code=species_code,
                                                subgroup_code=subgroup_code,
                                                src=src,
                                                loc_id=loc_id,
                                                )
        return season

    def cache_data(self, verbose: bool = True, desc: str = None) -> None:
        """
        Load all to-be-cached data in memory by iterating over the dataset
        While iterating the data is cached
        :param verbose: specifies whether a progress bar should be displayed
        :param desc: Progress bar description
        """
        if verbose:
            if desc is None:
                desc = 'Caching data'
            for _ in tqdm(self.iter_items(),
                          total=len(self),
                          desc=desc,
                          ):
                pass
        else:
            for _ in self.iter_items():
                pass

    @property
    def step(self) -> str:
        return self._step

    @staticmethod
    def _from_base(dataset: BaseDataset, **kwargs) -> 'Dataset':
        return Dataset(
            dataset._df_y,
            dataset._df_y_loc,
            **kwargs,
        )

    @staticmethod
    def load(key: str, **kwargs) -> 'Dataset':
        return Dataset._from_base(BaseDataset.load(key),
                                  debug_mode=key == 'debug',
                                  **kwargs,
                                  )

    def _copy_from_base(self, dataset: BaseDataset) -> 'Dataset':
        return Dataset._from_base(dataset,
                                  data_openmeteo=self._data_om,
                                  calendar=self._calendar,
                                  debug_mode=self._debug_mode,
                                  step=self._step,
                                  data_keys=self._data_keys,
                                  )

    def select_locations(self, locations) -> 'Dataset':
        base = super().select_locations(locations)
        return self._copy_from_base(base)

    def select_years(self, years) -> 'Dataset':
        base = super().select_years(years)
        return self._copy_from_base(base)

    # def select_locations_by_country_codes(self, country_codes) -> 'Dataset':
    #     base = super().select_locations_by_country_codes(country_codes)
    #     return self._copy_from_base(base)

    def select_by_observation_requirement(self, obs_key) -> 'Dataset':
        base = super().select_by_observation_requirement(obs_key)
        return self._copy_from_base(base)

    def select_by_local_num_observations(self, num_observations: int, obs_key) -> 'Dataset':
        base = super().select_by_local_num_observations(num_observations, obs_key)
        return self._copy_from_base(base)

    def select_by_ixs(self, ixs: list) -> 'Dataset':
        base = super().select_by_ixs(ixs)
        return self._copy_from_base(base)

    # def select_by_position_ixs(self, ixs: list) -> 'Dataset':
    #     base = super().select_by_position_ixs(ixs)
    #     return self._copy_from_base(base)

    def aggregate_in_grid(self, method: str = 'mean', grid_size: tuple = None) -> 'Dataset':
        base = super().aggregate_in_grid(method=method, grid_size=grid_size)
        return self._copy_from_base(base)

    def compute_feature_stats(self, verbose: bool = True) -> dict:
        """
        Computes the mean and standard deviation of each feature in the dataset.
        :return: a dict mapping all feature keys to a tuple containing the mean and standard deviation.
        """

        if self._debug_mode:  # If debug_mode -> skip the costly statistic computation
            return {k: (0, 1) for k in self._data_keys}

        stats = defaultdict(list)
        # Collect all values of all features
        if verbose:
            iter_samples = tqdm(self.iter_items(), total=len(self), desc='Computing feature statistics')
        else:
            iter_samples = self.iter_items()

        for sample in iter_samples:
            fs = sample[config.KEY_FEATURES]
            for key, f in fs.items():
                stats[key].append(f)
        # Concatenate them to one array
        stats = {
            key: np.concatenate(f) for key, f in stats.items()
        }
        # Compute statistics on the arrays
        stats = {
            key: (np.mean(f), np.std(f)) for key, f in stats.items()
        }
        # print(stats)
        return stats


if __name__ == '__main__':

    from tqdm import tqdm
    import time

    # _dataset = Dataset.load('debug',

    _time_start = time.time()

    # _dataset = Dataset.load('CPF_PEP725_winter_wheat',
    #                         mode_download_meteo='skip',
    #                         step='hourly',
    #                         save_meteo_data=True,
    #                         data_keys=('temperature_2m',
    #                                    'is_day',
    #                                    'precipitation',)
    #                         )
    #
    # print('%s seconds', time.time() - _time_start)
    # _time_start = time.time()
    #
    # print('Iter 1')
    # for _x in tqdm(_dataset.iter_items(), total=len(_dataset)):
    #     pass
    # print('Iter 2')
    # for _x in tqdm(_dataset.iter_items(), total=len(_dataset)):
    #     pass

    # _dataset.close()
    _dataset = Dataset.load('CPF_PEP725_winter_wheat',
                            # mode_download_meteo='skip',
                            step='hourly',
                            data_keys=('temperature_2m',
                                       'is_day',
                                       'precipitation',)
                            )
    print('seconds: ', time.time() - _time_start)

    # print(_dataset.compute_feature_stats(verbose=True))

    # _dates = set()
    #
    # for _x in _dataset.iter_items():
    #     # print(_x)
    #     print(_x[config.KEY_FEATURES][config.KEY_FEATURE_ISDAY].shape)
    #     _dates.add(_x['season_start'])
    #     input()
    # print(len(_dates))
    # print(_dates)

    print('Iter 1')
    for _x in tqdm(_dataset.iter_items(), total=len(_dataset)):
        pass
    print('Iter 2')
    for _x in tqdm(_dataset.iter_items(), total=len(_dataset)):
        pass

