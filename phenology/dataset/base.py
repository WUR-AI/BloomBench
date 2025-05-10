import os
from datetime import datetime
from functools import reduce, lru_cache

from collections import defaultdict
from typing import Iterator

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import linear_model

from phenology import config
from phenology.data.resources.load_admin_boundaries import load_admin_boundaries
from phenology.dataset.preprocessing.pep725 import get_pep725_dataframes
from phenology.dataset.preprocessing.gmu_cherry import get_gmu_cherry_dataset_japan, get_gmu_cherry_dataset_switzerland, get_gmu_cherry_dataset_south_korea
from phenology.dataset.util import DatasetException
from phenology.dataset.util.func_pandas import select_location, select_year, empty_df_like

from phenology.util.func import round_partial


class BaseDataset:

    def __init__(self,
                 df_y: pd.DataFrame,
                 df_y_loc: pd.DataFrame,
                 ):
        """
        Create a dataset for conveniently accessing phenology observations
        :param df_y: a dataframe containing phenology observations
                     the dataframe should be indexed by (data src, loc_id, year, species_code, subgroup_code, obs_type)
                     the dataframe should have a column containing observations
        :param df_y_loc: a dataframe containing location data
        """

        self._df_y = df_y
        self._df_y_loc = df_y_loc
        self._validate_dfs()

        # Sort data for faster lookups
        self._df_y.sort_index(inplace=True)
        self._df_y_loc.sort_index(inplace=True)

        # Create a shared index between all dataframes
        self._index = self._reset_index()

    def __len__(self):
        return len(self._index)

    def __contains__(self, index):
        return index in self._index

    @lru_cache(maxsize=None)  # Cache least-recently-used/requested data
    def __getitem__(self, index) -> dict:
        if isinstance(index, int):
            index = self._index[index]
            src, loc_id, year, species_code, subgroup_code = index

        elif isinstance(index, tuple):
            assert len(index) == len(config.KEYS_INDEX)
            src, loc_id, year, species_code, subgroup_code = index

        else:
            raise DatasetException(f'Unsupported index type {type(index)}')

        df_obs = self._df_y.loc[src, loc_id, year, species_code, subgroup_code]
        observations = df_obs.to_dict()[config.KEY_OBSERVATIONS]

        return {
            config.KEY_DATA_SOURCE: src,
            config.KEY_LOC_ID: loc_id,
            config.KEY_YEAR: year,
            config.KEY_SPECIES_CODE: species_code,
            config.KEY_SUBGROUP_CODE: subgroup_code,
            config.KEY_OBSERVATIONS: observations,
        }

    def _validate_dfs(self) -> None:
        pass  # TODO -- validate data frames

    def _reset_index(self) -> pd.Index:

        # Group observations by observation type (i.e. which phenological stage was observed)
        groups = self._df_y.groupby(level=config.KEY_OBS_TYPE)

        # Create an empty index before calling reduce (in case the dataset is empty)
        eix = pd.MultiIndex.from_tuples([], names=(
            config.KEY_DATA_SOURCE,
            config.KEY_LOC_ID,
            config.KEY_YEAR,
            config.KEY_SPECIES_CODE,
            config.KEY_SUBGROUP_CODE,
        ))

        # Take the union of the indices of all groups
        self._index = reduce(pd.MultiIndex.union,
                             [group.xs(key, level=config.KEY_OBS_TYPE).index for key, group in groups],
                             eix,
                             ).drop_duplicates(keep='first')

        return self._index.sortlevel(level=0, sort_remaining=True)[0]

    def iter_index(self) -> Iterator[tuple]:
        for i in self._index:
            yield i

    def iter_items(self) -> Iterator[dict]:
        for i in self.iter_index():
            yield self[i]

    @property
    def locations(self) -> list:
        locations = set(zip(self._index.get_level_values(config.KEY_DATA_SOURCE),
                            self._index.get_level_values(config.KEY_LOC_ID)))
        return list(locations)

    @property
    def species(self) -> list:
        species = set(zip(self._index.get_level_values(config.KEY_DATA_SOURCE),
                          self._index.get_level_values(config.KEY_SPECIES_CODE),
                          self._index.get_level_values(config.KEY_SUBGROUP_CODE),
                          ))
        return list(species)

    @property
    def years(self) -> list:
        return list(sorted(set(self._index.get_level_values(config.KEY_YEAR))))

    @property
    def num_observation_types(self) -> int:
        return self._df_y.index.get_level_values(level=config.KEY_OBS_TYPE).nunique()

    @property
    def observation_types(self) -> list:
        return self._df_y.index.get_level_values(level=config.KEY_OBS_TYPE).unique()

    # @property
    # def country_codes(self) -> list:
    #     locations = self.locations
    #     ccs = {self.get_location_country(loc)[0] for loc in locations}
    #     return list(ccs)

    @property
    def bounding_box(self):

        min_lat = np.inf
        max_lat = -np.inf
        min_lon = np.inf
        max_lon = -np.inf

        for i in self.iter_index():
            coords = self.get_location_coordinates(i, from_index=True)
            lon = coords[config.KEY_LON]
            lat = coords[config.KEY_LAT]

            if min_lat > lat:
                min_lat = lat
            if max_lat < lat:
                max_lat = lat
            if min_lon > lon:
                min_lon = lon
            if max_lon < lon:
                max_lon = lon

        return {
            'min_lat': min_lat,
            'max_lat': max_lat,
            'min_lon': min_lon,
            'max_lon': max_lon,
        }

    def get_location_coordinates(self, i: tuple, from_index: bool = False) -> dict:
        if from_index:
            src, loc_id, _, _, _ = i
        else:
            src, loc_id = i
        lat = self._df_y_loc.loc[src, loc_id][config.KEY_LAT]
        lon = self._df_y_loc.loc[src, loc_id][config.KEY_LON]
        return {
            'lat': lat,
            'lon': lon,
        }

    def get_location_name(self, i: tuple, from_index: bool = False) -> str:
        if from_index:
            src, loc_id, _, _, _ = i
        else:
            src, loc_id = i
        return self._df_y_loc.loc[src, loc_id][config.KEY_LOC_NAME]

    # def get_location_country(self, i: tuple, from_index: bool = False) -> tuple:
    #     if from_index:
    #         src, loc_id, _, _, _ = i
    #     else:
    #         src, loc_id = i
    #
    #     cc = self._df_y_loc.loc[src, loc_id][config.KEY_COUNTRY_CODE]
    #
    #     return cc, config.COUNTRY_CODE_NAME[cc]

    def select_locations(self, locations) -> 'BaseDataset':
        """
        Create a new BaseDataset with only the specified locations.
        :param locations: the species to select. Can be specified using either:
            - a two-tuple containing (data source key, loc_id)
            - a list of two-tuples (for selecting multiple locations)
        :return: a new BaseDataset object containing only the specified locations.
        """
        return BaseDataset(
            select_location(self._df_y, locations),
            self._df_y_loc,
        )

    def select_years(self, years) -> 'BaseDataset':
        """
        Create a new BaseDataset with only the specified years.
        :param years: the species to select. Can be specified using either:
            - a year (as integer)
            - a list of years
        :return: a new BaseDataset object containing only the specified years.
        """
        return BaseDataset(
            select_year(self._df_y, years),
            self._df_y_loc,
        )

    # def select_locations_by_country_codes(self, country_codes) -> 'BaseDataset':
    #     assert isinstance(country_codes, str) or isinstance(country_codes, list)
    #
    #     if isinstance(country_codes, str):
    #         locations = [
    #             loc for loc in self.locations if self.get_location_country(loc, from_index=False)[0] == country_codes
    #         ]
    #     else:
    #         locations = [
    #             loc for loc in self.locations if self.get_location_country(loc, from_index=False)[0] in country_codes
    #         ]
    #     return self.select_locations(locations)

    def select_by_local_num_observations(self, num_observations: int, obs_key) -> 'BaseDataset':
        assert num_observations >= 0
        if num_observations == 0:
            return BaseDataset(
                self._df_y,
                self._df_y_loc,
            )

        # df_y is indexed by
        # (data src, loc_id, year, species_code, subgroup_code, obs_type)

        df_y = self._df_y.groupby(
            level=[0, 1, 3, 4]
        ).filter(
            lambda df: len(df.xs(obs_key, level=config.KEY_OBS_TYPE)) >= num_observations
        )

        return BaseDataset(
            df_y,
            self._df_y_loc,
        )

    def select_by_observation_requirement(self, obs_key) -> 'BaseDataset':

        if not isinstance(obs_key, list):
            obs_key = [obs_key]

        ixs_selection = []
        for ix in self.iter_index():

            if all([(ix + (key,)) in self._df_y.index for key in obs_key]):
                ixs_selection.extend([(ix + (key,)) for key in obs_key])

        df_y = self._df_y.loc[ixs_selection]

        return BaseDataset(
            df_y,
            self._df_y_loc,
        )

    def select_by_ixs(self, ixs: list) -> 'BaseDataset':
        assert isinstance(ixs, list)

        # df_y is indexed by
        # (data src, loc_id, year, species_code, subgroup_code, obs_type)

        if len(ixs) == 0:
            return BaseDataset(
                empty_df_like(self._df_y),
                self._df_y_loc,
            )

        df_y = pd.concat([
            self._df_y.xs((src, loc_id, year, spc, sub),
                       level=[
                           config.KEY_DATA_SOURCE,
                           config.KEY_LOC_ID,
                           config.KEY_YEAR,
                           config.KEY_SPECIES_CODE,
                           config.KEY_SUBGROUP_CODE,
                       ],
                       drop_level=False,
                       )
            for src, loc_id, year, spc, sub in ixs
        ])

        return BaseDataset(
            df_y,
            self._df_y_loc,
        )

    # def select_by_position_ixs(self, ixs: list) -> 'BaseDataset':
    #     assert isinstance(ixs, list)
    #     # ixs do not match index of self._df_y and should thus be converted
    #     ixs = list(self._index[ixs].values)
    #
    #     # df_y is indexed by
    #     # (data src, loc_id, year, species_code, subgroup_code, obs_type)
    #
    #     df_y = pd.concat([
    #         self._df_y.xs((src, loc_id, year, spc, sub),
    #                    level=[
    #                        config.KEY_DATA_SOURCE,
    #                        config.KEY_LOC_ID,
    #                        config.KEY_YEAR,
    #                        config.KEY_SPECIES_CODE,
    #                        config.KEY_SUBGROUP_CODE,
    #                    ],
    #                    drop_level=False,
    #                    )
    #         for src, loc_id, year, spc, sub in ixs
    #     ])
    #
    #     return BaseDataset(
    #         df_y,
    #         self._df_y_loc,
    #     )

    def aggregate_in_grid(self,
                          method: str = 'median',
                          grid_size: tuple = None,
                          ) -> 'BaseDataset':
        """
        Divide the dataset into grid cells. Select one observation per grid cell.
        :param method: Method of aggregation/selection. 'median' by default.
        :param grid_size: tuple of (grid latitude cell size, grid longitude cell size)
        :return:
        """

        # Select minimal grid size if none was provided
        grid_size = grid_size or config.MIN_GRID_SIZE

        res_lat, res_lon = grid_size

        # Create a copy of the observations dataframe
        # Add a latitude and longitude column
        df_selection = self._df_y.join(self._df_y_loc[[config.KEY_LAT, config.KEY_LON]])

        # Round location coordinates to corresponding grid cell
        df_selection[config.KEY_LAT] = df_selection[config.KEY_LAT].apply(lambda x: round_partial(x, res_lat))
        df_selection[config.KEY_LON] = df_selection[config.KEY_LON].apply(lambda x: round_partial(x, res_lon))

        # Subsequent steps depend on aggregation method
        match method:
            case 'mean':

                raise Exception('Deprecated: First mean method of aggregation should be fixed')

                # df_selection[config.KEY_OBSERVATIONS] = df_selection[config.KEY_OBSERVATIONS].astype(np.int64)
                #
                # df_selection.reset_index(inplace=True)
                #
                # # Aggregate the phenology observations
                # df = df_selection.groupby(
                #     [config.KEY_LAT, config.KEY_LON, config.KEY_OBS_TYPE, config.KEY_YEAR],
                #     as_index=False,
                # ).agg(
                #     {
                #         config.KEY_LAT: 'first',
                #         config.KEY_LON: 'first',
                #         config.KEY_OBS_TYPE: 'first',
                #         config.KEY_YEAR: 'first',
                #         config.KEY_OBSERVATIONS: 'mean',
                #         config.KEY_DATA_SOURCE: 'first',
                #         config.KEY_LOC_ID: 'first',
                #         config.KEY_SPECIES_CODE: 'first',
                #         config.KEY_SUBGROUP_CODE: 'first',
                #     }
                # )
                #
                # df.reset_index(inplace=True)
                #
                # df[config.KEY_OBSERVATIONS] = pd.to_datetime(df[config.KEY_OBSERVATIONS]).apply(lambda x: np.datetime64(x.date()))
                #
                # df.set_index(list(config.KEYS_INDEX + (config.KEY_OBS_TYPE,)), inplace=True)
                # df.drop([config.KEY_LAT, config.KEY_LON, 'index'], axis=1, inplace=True)
                #
                # return BaseDataset(
                #     df_y=df,
                #     df_y_loc=self._df_y_loc,
                # )

            case 'median':

                # df_selection[config.KEY_OBSERVATIONS] = df_selection[config.KEY_OBSERVATIONS].astype(np.int64)

                df_selection.reset_index(inplace=True)

                # Aggregate the phenology observations
                df = df_selection.groupby(
                    [config.KEY_LAT, config.KEY_LON, config.KEY_OBS_TYPE, config.KEY_YEAR],
                    as_index=False,
                ).agg(
                    {
                        config.KEY_LAT: 'first',
                        config.KEY_LON: 'first',
                        config.KEY_OBS_TYPE: 'first',
                        config.KEY_YEAR: 'first',
                        config.KEY_OBSERVATIONS: 'median',
                        config.KEY_DATA_SOURCE: 'first',
                        config.KEY_LOC_ID: 'first',
                        config.KEY_SPECIES_CODE: 'first',
                        config.KEY_SUBGROUP_CODE: 'first',
                    }
                )

                df.reset_index(inplace=True)

                # df[config.KEY_OBSERVATIONS] = pd.to_datetime(df[config.KEY_OBSERVATIONS]).apply(
                #     lambda x: np.datetime64(x.date()))

                df.set_index(list(config.KEYS_INDEX + (config.KEY_OBS_TYPE,)), inplace=True)
                df.drop([config.KEY_LAT, config.KEY_LON, 'index'], axis=1, inplace=True)

                return BaseDataset(
                    df_y=df,
                    df_y_loc=self._df_y_loc,
                )

            case 'first':
                raise NotImplementedError  # TODO
            case _:
                raise DatasetException(f'Unknown aggregation method "{method}" for phenological observations')

    def split_by_grid(self,
                      grid_size: tuple,
                      split_size: float,
                      shuffle: bool = True,
                      random_state: int = None,
                      ) -> tuple:
        """
        Assign each datapoint to a cell in a spatial grid.
        Randomly assign cells and corresponding data to two new datasets.
        :param grid_size: size of the grid cells (in degrees)
        :param split_size: size (float) of the data split
                           Proportion of cells assigned to the first dataset
                           Remainder is assigned to the second dataset
        :param shuffle: whether to shuffle the cells before splitting
        :param random_state: the random seed for shuffling the cells
        :return: a three-tuple containing (first dataset, the second dataset, info dict about the split).
        """
        assert 0 <= split_size <= 1
        lat_size, lon_size = grid_size

        cell_to_locations = defaultdict(set)

        # Assign locations to cells by discretizing their coordinates
        for loc in self.locations:
            coords = self.get_location_coordinates(loc, from_index=False)
            lat = coords[config.KEY_LAT]
            lon = coords[config.KEY_LON]
            cell = (round_partial(lat, lat_size), round_partial(lon, lon_size))
            cell_to_locations[cell].add(loc)

        # Get all cells containing data
        cells = list(cell_to_locations.keys())
        # Sort cells to ensure determinism in split
        cells = sorted(cells)

        # Split the cells
        cells_1, cells_2 = train_test_split(cells,
                                            train_size=split_size,
                                            random_state=random_state,
                                            shuffle=shuffle,
                                            )

        # Merge locations corresponding to the cell split
        locs_1 = set.union(*[cell_to_locations[cell] for cell in cells_1])
        locs_2 = set.union(*[cell_to_locations[cell] for cell in cells_2])
        # Ensure there's no overlap -- as sanity check
        assert len(set.intersection(locs_1, locs_2)) == 0

        # Return two datasets with only the selected locations
        return (self.select_locations(list(locs_1)),
                self.select_locations(list(locs_2)),
                {
                    'cells_1': cells_1,
                    'cells_2': cells_2,
                    'cells_to_locations': cell_to_locations,
                    'cell_size': grid_size,
                },
                )

    def observation_counts(self) -> dict:
        counts = self._df_y.value_counts(subset=config.KEY_OBS_TYPE)
        return counts.to_dict()

    """
    ####################################################################################################################
     View
    ####################################################################################################################
    """

    def savefig_observation_hists(self, path: str, n_bins: int = 50) -> None:  # TODO -- hist_kwargs

        fig, ax = plt.subplots(nrows=self.num_observation_types, sharex=True, sharey=True)
        if self.num_observation_types == 1:
            ax = (ax,)
        for i, obs_type in enumerate(self._df_y.index.get_level_values(level=config.KEY_OBS_TYPE).unique()):
            hist = self._df_y.xs(obs_type,
                                 level=config.KEY_OBS_TYPE,
                                 ).apply(lambda x: x.dt.dayofyear).hist(bins=n_bins, sharex=True, sharey=True, ax=ax[i])
            # ax[i].set_title(f'Observation type: {obs_type}')
            ax[i].set_title('')
            ax[i].set_xlim(0, 365)
            ax[i].set_xlabel(f'Day of year')
            # ax[i].set_ylabel(f'Count')
            ax[i].set_ylabel(f'# {obs_type}')

        fn = 'histograms.png'

        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, fn), bbox_inches='tight')
        plt.close()

    def savefig_observation_over_time(self, path: str):  # TODO -- agg per year

        fig, axs = plt.subplots(nrows=self.num_observation_types, sharex=True, sharey=False)

        x_obs = defaultdict(list)
        y_obs = defaultdict(list)

        for x in self.iter_items():

            for obs_type, obs in x[config.KEY_OBSERVATIONS].items():
                x_obs[obs_type].append(x[config.KEY_YEAR])
                y_obs[obs_type].append(obs.dayofyear)

        for ax, obs_type in zip(axs, self.observation_types):

            y_min = min(y_obs[obs_type])
            y_max = max(y_obs[obs_type])

            ax.scatter(x_obs[obs_type],
                       y_obs[obs_type],
                       s=1,
                       alpha=0.5,
                       )
            ax.set_title(f'Observation type: {obs_type}')
            ax.set_ylim(y_min, y_max)

        os.makedirs(path, exist_ok=True)
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def savefig_observation_mean_over_time(self,
                                           path: str,
                                           obs_type: str,
                                           dpi: int = 500,
                                           ):

        xs = []  # years
        ys = []  # mean observation
        counts = self.observation_counts()
        if obs_type in counts.keys() and counts[obs_type] > 0:
            df_y = self._df_y.xs(obs_type, level=config.KEY_OBS_TYPE)
            df_y = df_y.groupby(config.KEY_YEAR).mean()
            xys = df_y.to_dict()[config.KEY_OBSERVATIONS]

            for x, y in xys.items():
                xs.append(x)
                ys.append(y.dayofyear)

        # Fit linear trend
        reg = linear_model.LinearRegression()
        reg.fit(np.array(xs).reshape(-1, 1), ys)
        xs_lin = list(range(min(xs), max(xs) + 1))
        ys_lin = reg.predict(np.array(xs_lin).reshape(-1, 1))

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        ax.scatter(xs, ys, color='tomato')
        a = reg.coef_[0]
        b = reg.intercept_
        ax.plot(xs_lin, ys_lin, '--', color='black', label=f'$f(x)=${a:.2f}$x${'+' if b >= 0 else ''}{b:.2f}')

        ymin = 0
        ymax = 365
        if len(ys) > 0:
            margin = 1.
            dy = round((max(ys) - min(ys)) * margin)
            ymin = max(ymin, min(ys) - dy)
            ymax = min(ymax, max(ys) + dy)

        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('Year')
        ax.set_ylabel(f'Day of year ({obs_type})')

        ax.legend()

        # Save figure
        path_svg = os.path.join(path, 'svg')
        path_png = os.path.join(path, 'png')

        os.makedirs(path_svg, exist_ok=True)
        os.makedirs(path_png, exist_ok=True)

        plt.savefig(os.path.join(path_svg, f'observation_mean_over_time.svg'),
                    bbox_inches='tight',
                    )
        plt.savefig(os.path.join(path_png, f'observation_mean_over_time.png'),
                    bbox_inches='tight',
                    dpi=dpi,
                    )
        plt.close()

    def savefig_observation_map(self,
                                path: str,
                                dpi: int = 500,
                                ):
        """
        Scatter all observations in this dataset on a map and save it
        Map is saved as png file and svg file
        :param path: where to save the map
        :param dpi: dpi of the .png file
        :return:
        """

        # Get all observation types present
        obs_types = self.observation_types

        # Load administrative boundaries for plotting
        gdf_admin = load_admin_boundaries()
        # Get dataset spatial bounding box to filter administrative region
        bb = self.bounding_box

        minx = bb['min_lon']
        miny = bb['min_lat']
        maxx = bb['max_lon']
        maxy = bb['max_lat']

        gdf_admin = gdf_admin.cx[minx:maxx, miny:maxy]

        # Group all observations of the specific year by observation type
        obs_per_type = {
            ot: list() for ot in obs_types
        }

        for ot in obs_types:  # TODO -- option to color by obs type -- or color by species subgroup!
            # Get all observations for the specific year and observation type
            # Drop observation type from index
            # df index: (data src, loc_id, year, species_code, subgroup_code)
            df_year_ot = self._df_y.xs(ot, level=config.KEY_OBS_TYPE, drop_level=True)

            # Iterate over samples in the dataframe. Store their coordinates
            for i in df_year_ot.index:
                coords = self.get_location_coordinates(i, from_index=True)
                coords = (coords[config.KEY_LON], coords[config.KEY_LAT])
                obs_per_type[ot].append(coords)

        # transform coordinate lists in geopandas dataframes
        obs_per_type = {
            ot: gpd.GeoDataFrame(
                obs,
                geometry=gpd.points_from_xy(
                    [x for x, y in obs],
                    [y for x, y in obs],
                ),
                crs=gdf_admin.crs,
            ) for ot, obs in obs_per_type.items()
        }

        # Plot observations per observation type
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        # First plot country borders
        gdf_admin.plot(ax=ax, color='lightgrey', edgecolor='black')

        for ot, gdf_obs in obs_per_type.items():
            gdf_obs.plot(ax=ax,
                         label=ot,
                         marker='o',
                         markersize=0.6,
                         alpha=0.5,
                         color='tomato',
                         )

        # Set bounds to map
        margin = 0.1  # Keep a small margin bordering the plot as a percentage of total distance
        x_margin = (maxx - minx) * margin
        y_margin = (maxy - miny) * margin
        ax.set_xlim(minx - x_margin, maxx + x_margin)
        ax.set_ylim(miny - y_margin, maxy + x_margin)

        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')

        # Save figure

        path_svg = os.path.join(path, 'svg')
        path_png = os.path.join(path, 'png')

        os.makedirs(path_svg, exist_ok=True)
        os.makedirs(path_png, exist_ok=True)
        plt.savefig(os.path.join(path_svg, f'observations_map.svg'),
                    bbox_inches='tight',
                    )
        plt.savefig(os.path.join(path_png, f'observations_map.png'),
                    bbox_inches='tight',
                    dpi=dpi,
                    )
        plt.close()

    def savefigs_observation_maps_over_time(self,
                                            path: str,
                                            verbose: bool = True,
                                            dpi: int = 500,
                                            ):
        """
        For every year in this dataset, generate a map containing scatter plots of observation locations
        Points are colored by observation type
        Plots are saved both as .svg and .png files
        :param path: where to store the plots
        :param verbose: whether to print progression
        :param dpi: dpi of png plots
        """

        # Figures will be created per year
        iter_years = self.years
        if verbose:
            iter_years = tqdm(iter_years, total=len(iter_years), desc='Creating observations maps...')

        # Get all observation types present
        obs_types = self.observation_types

        # Load administrative boundaries for plotting
        gdf_admin = load_admin_boundaries()
        # Get dataset spatial bounding box to filter administrative region
        bb = self.bounding_box

        minx = bb['min_lon']
        miny = bb['min_lat']
        maxx = bb['max_lon']
        maxy = bb['max_lat']

        gdf_admin = gdf_admin.cx[minx:maxx, miny:maxy]

        for year in iter_years:

            # Get all observations for the specific year
            # Keep year in index
            # df index: (data src, loc_id, year, species_code, subgroup_code, obs_type)
            df_year = self._df_y.xs(year, level=config.KEY_YEAR, drop_level=False)

            # Group all observations of the specific year by observation type
            obs_per_type = {
                ot: list() for ot in obs_types
            }

            for ot in obs_types:
                # Get all observations for the specific year and observation type
                # Drop observation type from index
                # df index: (data src, loc_id, year, species_code, subgroup_code)

                if ot not in df_year.index.get_level_values(config.KEY_OBS_TYPE):
                    continue

                df_year_ot = df_year.xs(ot, level=config.KEY_OBS_TYPE, drop_level=True)

                # Iterate over samples in the dataframe. Store their coordinates
                for i in df_year_ot.index:
                    coords = self.get_location_coordinates(i, from_index=True)
                    coords = (coords[config.KEY_LON], coords[config.KEY_LAT])
                    obs_per_type[ot].append(coords)

            # transform coordinate lists in geopandas dataframes
            obs_per_type = {
                ot: gpd.GeoDataFrame(
                    obs,
                    geometry=gpd.points_from_xy(
                        [x for x, y in obs],
                        [y for x, y in obs],
                    ),
                    crs=gdf_admin.crs,
                ) for ot, obs in obs_per_type.items()
            }

            # Plot observations per observation type
            fig, ax = plt.subplots(1, 1)
            # First plot country borders
            gdf_admin.plot(ax=ax, color='lightgrey', edgecolor='black')

            for ot, gdf_obs in obs_per_type.items():
                if len(gdf_obs) == 0:
                    continue
                gdf_obs.plot(ax=ax,
                             label=ot,
                             marker='o',
                             markersize=1,
                             alpha=0.1,
                             )

            # Set bounds to map
            margin = 0.1  # Keep a small margin bordering the plot as a percentage of total distance
            x_margin = (maxx - minx) * margin
            y_margin = (maxy - miny) * margin
            ax.set_xlim(minx - x_margin, maxx + x_margin)
            ax.set_ylim(miny - y_margin, maxy + x_margin)

            # Save figure

            path_svg = os.path.join(path, 'svg')
            path_png = os.path.join(path, 'png')

            os.makedirs(path_svg, exist_ok=True)
            os.makedirs(path_png, exist_ok=True)
            plt.savefig(os.path.join(path_svg, f'observations_map_{year}.svg'),
                        bbox_inches='tight',
                        )
            plt.savefig(os.path.join(path_png, f'observations_map_{year}.png'),
                        bbox_inches='tight',
                        dpi=dpi,
                        )
            plt.close()

        # TODO -- generate video

    def savefig_species_subgroup_occurrence_temporal(self, path: str, dpi: int = 500):

        sss_year_occurrence = defaultdict(set)

        # df index: (data src, loc_id, year, species_code, subgroup_code)
        for src, _, year, species, subgroup in self.iter_index():
            sss_year_occurrence[src, species, subgroup].add(year)

        fig, ax = plt.subplots(1, 1,
                               # figsize=(10, 3 + len(sss_year_occurrence.keys())),
                               )

        bar_height = 0.4
        y = -bar_height / 2
        labels = []
        bars = []
        for (src, species, subgroup), years in sss_year_occurrence.items():

            # broken_barh(xranges, (ymin, height))
            # xranges is a sequence of (start, duration) tuples
            bar = ax.broken_barh([(year, 1) for year in years], (y, bar_height),
                                 color='grey',
                                 )

            bars.append(bar)
            y += 1
            labels.append((species, subgroup))

        ax.set_yticks(range(len(labels)),
                      labels=labels)
        ax.set_xlabel('Year')
        ax.set_ylabel('(Species ID, subgroup ID) occurrence')

        # # Access each bar's segments and adjust their heights if necessary
        # for bar in bars:
        #     for patch in bar.get_children():
        #         y, h = patch.get_y(), patch.get_height()
        #         if h > bar_height:
        #             # Cap the height at max_height and reset y to 0 (assuming it starts from bottom)
        #             patch.set_height(bar_height)
        #             patch.set_y(0)
        #         else:
        #             # Keep original values
        #             pass

        # Save figure
        path_svg = os.path.join(path, 'svg')
        path_png = os.path.join(path, 'png')

        os.makedirs(path_svg, exist_ok=True)
        os.makedirs(path_png, exist_ok=True)
        plt.savefig(os.path.join(path_svg, f'species_subgroup_occurrence.svg'),
                    bbox_inches='tight',
                    )
        plt.savefig(os.path.join(path_png, f'species_subgroup_occurrence.png'),
                    bbox_inches='tight',
                    dpi=dpi,
                    )
        plt.close()

    """
    ####################################################################################################################
     Pre-defined datasets
    ####################################################################################################################
    """

    @staticmethod
    def load(key: str) -> 'BaseDataset':
        """
        Load a pre-configured dataset based on its key/name

        :param key: the key/name of the dataset to load
        """

        """
        CPF Config
        """
        # cpf_year_min = 1980
        cpf_year_min = 1986
        cpf_year_max = 2015
        cpf_year_range = list(range(cpf_year_min, cpf_year_max + 1))
        cpf_remove_outliers = True
        # cpf_remove_outliers = False
        cpf_do_agg = True
        # cpf_do_agg = False
        # cpf_agg_method = 'mean'
        cpf_agg_method = 'median'

        """
        Benchmark config
        """
        # Set year range
        bm_year_min = 1980  # Start year
        bm_year_max = datetime.now().year - 1  # Set previous year as start year
        bm_years = list(range(bm_year_min, bm_year_max + 1))
        bm_do_agg = True
        bm_agg_method = 'median'
        bm_remove_outliers = True
        bm_assert_target = True

        match key:

            case 'test_dataset':

                dfs = get_pep725_dataframes(
                    filter_on_species=(config.KEY_PEP725, 333, 300),
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=cpf_remove_outliers,
                    filter_on_observation_types=['BBCH_0', 'BBCH_51'],
                    filter_on_years=cpf_year_range,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                # print(df_y)
                # print(df_y_loc)

                df_y = BaseDataset._modify_year(df_y, 'BBCH_0', 1)  # TODO -- this is a hack

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                base = base.select_by_observation_requirement(['BBCH_0', 'BBCH_51'])

                if cpf_do_agg:
                    base = base.aggregate_in_grid(method=cpf_agg_method)

                return base


            # # PEP725 winter wheat
            # case 'CPF_PEP725_winter_wheat':
            #
            #     dfs = get_pep725_dataframes(
            #         filter_on_species=(config.KEY_PEP725, 333, 300),
            #         aggregation_method=None,  # Aggregate after preprocessing
            #         remove_outliers=cpf_remove_outliers,
            #         filter_on_observation_types=['BBCH_0', 'BBCH_51'],
            #         filter_on_years=cpf_year_range,
            #         datetime_observations=True,
            #     )
            #
            #     df_y = dfs['data']
            #     df_y_loc = dfs['locations']
            #
            #     # print(df_y)
            #     # print(df_y_loc)
            #
            #     df_y = BaseDataset._modify_year(df_y, 'BBCH_0', 1)  # TODO -- this is a hack
            #
            #     base = BaseDataset(
            #         df_y,
            #         df_y_loc,
            #     )
            #
            #     base = base.select_locations_by_country_codes('DE')
            #
            #     base = base.select_by_observation_requirement(['BBCH_0', 'BBCH_51'])
            #
            #     if cpf_do_agg:
            #         base = base.aggregate_in_grid(method=cpf_agg_method)
            #
            #     return base
            #
            # # PEP725 winter barley
            # case 'CPF_PEP725_winter_barley':
            #
            #     dfs = get_pep725_dataframes(
            #         filter_on_species=(config.KEY_PEP725, 330, 300),
            #         aggregation_method=None,  # Aggregate after preprocessing
            #         remove_outliers=cpf_remove_outliers,
            #         filter_on_observation_types=['BBCH_0', 'BBCH_51'],
            #         filter_on_years=cpf_year_range,
            #         datetime_observations=True,
            #     )
            #
            #     df_y = dfs['data']
            #     df_y_loc = dfs['locations']
            #
            #     df_y = BaseDataset._modify_year(df_y, 'BBCH_0', 1)  # TODO -- this is a hack
            #
            #     base = BaseDataset(
            #         df_y,
            #         df_y_loc,
            #     )
            #
            #     base = base.select_locations_by_country_codes('DE')
            #
            #     base = base.select_by_observation_requirement(['BBCH_0', 'BBCH_51'])
            #
            #     if cpf_do_agg:
            #         base = base.aggregate_in_grid(method=cpf_agg_method)
            #
            #     return base
            #
            # # PEP725 winter rye
            # case 'CPF_PEP725_winter_rye':
            #
            #     dfs = get_pep725_dataframes(
            #         filter_on_species=(config.KEY_PEP725, 332, 300),
            #         aggregation_method=None,  # Aggregate after preprocessing
            #         remove_outliers=cpf_remove_outliers,
            #         filter_on_observation_types=['BBCH_0', 'BBCH_61'],
            #         filter_on_years=cpf_year_range,
            #         datetime_observations=True,
            #     )
            #
            #     df_y = dfs['data']
            #     df_y_loc = dfs['locations']
            #
            #     df_y = BaseDataset._modify_year(df_y, 'BBCH_0', 1)  # TODO -- this is a hack
            #
            #     base = BaseDataset(
            #         df_y,
            #         df_y_loc,
            #     )
            #
            #     base = base.select_locations_by_country_codes('DE')
            #
            #     base = base.select_by_observation_requirement(['BBCH_0', 'BBCH_61'])
            #
            #     if cpf_do_agg:
            #         base = base.aggregate_in_grid(method=cpf_agg_method)
            #
            #     return base

            case 'GMU_Cherry_Japan':

                dfs = get_gmu_cherry_dataset_japan(
                    # remove_outliers=bm_remove_outliers,
                    remove_outliers=False,  # Set to false since multiple species occur in this dataset!
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                base = base.select_years(
                    years=bm_years,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('gmu_0')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'GMU_Cherry_Switzerland':

                dfs = get_gmu_cherry_dataset_switzerland(
                    remove_outliers=bm_remove_outliers,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                base = base.select_years(
                    years=bm_years,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('gmu_1')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'GMU_Cherry_South_Korea':

                dfs = get_gmu_cherry_dataset_south_korea(
                    remove_outliers=bm_remove_outliers,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                base = base.select_years(
                    years=bm_years,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('gmu_2')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Apple':  # Malus x Domestica

                species_subgroups = [
                    (config.KEY_PEP725, 220, 100),  # Early cultivar
                    (config.KEY_PEP725, 220, 130),  # Late cultivar
                    (config.KEY_PEP725, 220, 115),  # Middle cultivar
                    (config.KEY_PEP725, 220, 433),  # Cox Orange Renette
                    (config.KEY_PEP725, 220, 508),  # Elstar
                    (config.KEY_PEP725, 220, 437),  # Golden Delicious
                    (config.KEY_PEP725, 220, 430),  # Goldparm
                    (config.KEY_PEP725, 220, 438),  # Gravensteiner
                    (config.KEY_PEP725, 220, 509),  # Idared
                    (config.KEY_PEP725, 220, 500),  # James Grieve
                    (config.KEY_PEP725, 220, 501),  # Jonagold
                    (config.KEY_PEP725, 220, 510),  # Jonathan
                    (config.KEY_PEP725, 220, 503),  # Roter Boskoop
                    (config.KEY_PEP725, 220, 506),  # Wei
                    (config.KEY_PEP725, 220, 615),  # Granny Smith
                    (config.KEY_PEP725, 220, 617),  # Bobovec
                ]
                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60', 'BBCH_69', 'BBCH_87'],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Pear':  # Pyrus Communis

                species_subgroups = [
                    (config.KEY_PEP725, 227, 100),  # Early cultivar
                    (config.KEY_PEP725, 227, 130),  # Late cultivar
                    (config.KEY_PEP725, 227, 590),  # Williams
                    (config.KEY_PEP725, 227, 586),  # Bunte Julibirne
                    (config.KEY_PEP725, 227, 585),  # Jakob
                    (config.KEY_PEP725, 227, 587),  # Junsko Zlato
                    (config.KEY_PEP725, 227, 589),  # Karamanka
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60', 'BBCH_65', 'BBCH_69', 'BBCH_87'],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Peach':  # Prunus Persica

                species_subgroups = [
                    (config.KEY_PEP725, 202, 0),  # No group
                    (config.KEY_PEP725, 202, 579),  # Alberta
                    (config.KEY_PEP725, 202, 580),  # Dixired
                    (config.KEY_PEP725, 202, 581),  # Hale
                    (config.KEY_PEP725, 202, 578),  # Red Haven
                    (config.KEY_PEP725, 202, 582),  # Springtime
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60',],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Almond':  # Prunus Amygdalis

                species_subgroups = [
                    (config.KEY_PEP725, 782, 0),  # No group
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60', 'BBCH_65', 'BBCH_69', 'BBCH_87'],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Hazel':  # Corylus Avellana

                species_subgroups = [
                    (config.KEY_PEP725, 107, 0),  # No group
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60', 'BBCH_86',],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Cherry':  # Prunus Avium

                species_subgroups = [
                    (config.KEY_PEP725, 222, 0),    # No group
                    (config.KEY_PEP725, 222, 100),  # Early cultivar
                    (config.KEY_PEP725, 222, 130),  # Late cultivar
                    (config.KEY_PEP725, 222, 494),  # Regina
                    (config.KEY_PEP725, 222, 495),  # Schwarze Knorpelkirsch
                    (config.KEY_PEP725, 222, 618),  # Majska rana
                    (config.KEY_PEP725, 222, 602),  # Germersdorfer
                    (config.KEY_PEP725, 222, 603),  # Hedelfinger
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60', 'BBCH_65', 'BBCH_69', 'BBCH_87'],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Apricot':  # Prunus Armeniaca

                species_subgroups = [
                    (config.KEY_PEP725, 205, 0),  # No group
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60', 'BBCH_87', ],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Plum':  # Prunus Domestica

                species_subgroups = [
                    (config.KEY_PEP725, 225, 0),    # No group
                    (config.KEY_PEP725, 225, 100),  # Early cultivar
                    (config.KEY_PEP725, 225, 130),  # Late cultivar
                    (config.KEY_PEP725, 225, 621),  # Besztercei
                    (config.KEY_PEP725, 225, 595),  # Bosankska
                    (config.KEY_PEP725, 225, 596),  # Dzanarika
                    (config.KEY_PEP725, 225, 597),  # Pozegaca
                    (config.KEY_PEP725, 225, 612),  # Renkloda
                    (config.KEY_PEP725, 225, 614),  # Stanlay
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60', 'BBCH_65', 'BBCH_69', 'BBCH_87'],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Blackthorn':  # Prunus Spinosa

                species_subgroups = [
                    (config.KEY_PEP725, 123, 0),
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=['BBCH_60',],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case 'PEP725_Oak':  # Prunus Persica

                species_subgroups = [
                    (config.KEY_PEP725, 111, 0),  # No group
                ]

                dfs = get_pep725_dataframes(
                    filter_on_species=species_subgroups,
                    aggregation_method=None,  # Aggregate after preprocessing
                    remove_outliers=bm_remove_outliers,
                    filter_on_observation_types=[
                        'BBCH_11',
                        'BBCH_94',
                        'BBCH_86',
                        'BBCH_95',
                    ],
                    filter_on_years=bm_years,
                    datetime_observations=True,
                )

                df_y = dfs['data']
                df_y_loc = dfs['locations']

                base = BaseDataset(
                    df_y,
                    df_y_loc,
                )

                # Only keep data samples where the required observations are present
                if bm_assert_target:
                    base = base.select_by_observation_requirement('BBCH_60')
                # Aggregate observations in a grid
                if bm_do_agg:
                    base = base.aggregate_in_grid(method=bm_agg_method)

                return base

            case _:
                raise DatasetException(f'Undefined dataset key "{key}"')

    @staticmethod
    def _modify_year(df: pd.DataFrame, obs_type, offset):
        groups = df.groupby(level=config.KEY_OBS_TYPE)
        dfs = [group if key != obs_type else BaseDataset._modify_year_helper(group, offset) for key, group in groups]
        return pd.concat(dfs)

    @staticmethod
    def _modify_year_helper(df: pd.DataFrame, offset):
        df.reset_index(inplace=True)
        df[config.KEY_YEAR] = df[config.KEY_YEAR] + offset
        df.set_index(list(config.KEYS_INDEX + (config.KEY_OBS_TYPE,)), inplace=True)
        return df


if __name__ == '__main__':
    from tqdm import tqdm

    # _dataset_name = 'CPF_PEP725_winter_wheat'
    # _dataset_name = 'GMU_Cherry_Japan'
    # _dataset_name = 'GMU_Cherry_Switzerland'
    # _dataset_name = 'GMU_Cherry_South_Korea'
    # _dataset_name = 'PEP725_Oak'
    _dataset_name = 'PEP725_Apple'
    # _dataset_name = 'PEP725_Pear'
    # _dataset_name = 'PEP725_Peach'
    # _dataset_name = 'PEP725_Almond'
    # _dataset_name = 'PEP725_Hazel'
    # _dataset_name = 'PEP725_Cherry'
    # _dataset_name = 'PEP725_Apricot'
    # _dataset_name = 'PEP725_Blackthorn'
    # _dataset_name = 'PEP725_Plum'
    _dataset = BaseDataset.load(_dataset_name)

    # print(_dataset._index[0])

    # _dataset_selection = _dataset.select_by_observation_requirement([0, 51])

    _dataset_selection = _dataset.aggregate_in_grid()

    # _dataset_selection.show_observation_hists()
    # _dataset_selection.show_observation_over_time()

    # print(_dataset_selection.observation_counts())
    #
    # print(len(_dataset))
    # print(len(_dataset_selection))

    # _dataset.select_by_local_num_observations(2, 51)

    # for _x in _dataset_selection.iter_items():
    # for _x in tqdm(_dataset_selection.iter_items(), total=len(_dataset_selection)):
    #     # print(_x)
    #     pass
    # #     input()
    #
    # for _x in tqdm(_dataset_selection.iter_items(), total=len(_dataset_selection)):
    #     # print(_x)
    #     pass

    # _dataset.savefig_observation_map('temp_map')
    # _dataset.savefigs_observation_maps_over_time('temp_map_over_time')
    # _dataset.savefig_observation_hists('temp_hist')

    # _dataset.savefigs_complete(f'temp_dataset_figures/{_dataset_name}')
    #
    # _dataset.savefig_observation_mean_over_time('temp', 'BBCH_60')

    _dataset.savefig_species_subgroup_occurrence_temporal('temp')

    print(_dataset.observation_counts())

