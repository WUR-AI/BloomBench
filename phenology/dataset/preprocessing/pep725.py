import numpy as np
import pandas as pd

from phenology import config
from phenology.data.pep725.download import get_pep725_data
from phenology.dataset.util import DatasetException
from phenology.util.func import round_partial
from phenology.dataset.util.func_pandas import select_species, select_year, select_location, select_observation_type


"""
    Preprocess PEP725 into dataframes compatible with the dataset classes
"""

# Phenological observations are aggregated per (spatial) grid cell
_AGG_GRID = config.AGG_GRID_SIZE


def get_pep725_dataframes(aggregation_method: str = None,
                          remove_outliers: bool = True,
                          filter_on_species=None,
                          filter_on_years=None,
                          filter_on_locations=None,
                          filter_on_observation_types=None,
                          datetime_observations: bool = True,
                          ) -> dict:
    """
    Process PEP725 data to dataframes compatible with a TargetDataset object
    :return:
    """
    src = config.KEY_PEP725

    data_dict = get_pep725_data()

    """
    Prepare location dataframe
    """

    df_y_loc = data_dict['stations']

    df_y_loc[config.KEY_DATA_SOURCE] = src

    df_y_loc.reset_index(inplace=True)

    df_y_loc.set_index([config.KEY_DATA_SOURCE, 'PEP_ID', ], inplace=True)

    df_y_loc.index.names = (config.KEY_DATA_SOURCE, config.KEY_LOC_ID)

    df_y_loc.rename(
        columns={
            'LAT': config.KEY_LAT,
            'LON': config.KEY_LON,
            'NAME': config.KEY_LOC_NAME,
            'country_code': config.KEY_COUNTRY_CODE,
        },
        inplace=True,
    )

    """
    Prepare/format target dataframe
    """

    df_y = data_dict['data']

    df_y.reset_index(inplace=True)

    df_y[config.KEY_DATA_SOURCE] = src

    # Convert observation types to string
    df_y['BBCH'] = df_y['BBCH'].map(lambda x: f'BBCH_{x}')

    df_y.set_index([config.KEY_DATA_SOURCE, 'PEP_ID', 'YEAR', 'species_code', 'subgroup_code', 'BBCH'], inplace=True)

    df_y.index.names = config.KEYS_INDEX + (config.KEY_OBS_TYPE,)

    df_y.rename(
        columns={'DAY': config.KEY_OBSERVATIONS},
        inplace=True,
    )

    """
    Filter target dataframe
    """

    # print('Initial: ', len(df_y))

    # Optionally filter species
    if filter_on_species is not None:
        df_y = select_species(df_y, filter_on_species)

    # print('Filtered species: ', len(df_y))

    # Optionally filter years
    if filter_on_years is not None:
        df_y = select_year(df_y, filter_on_years)

    # print('Filtered years: ', len(df_y))

    # Optionally filter locations
    if filter_on_locations is not None:
        df_y = select_location(df_y, filter_on_locations)

    # print('Filtered locations: ', len(df_y))

    # Optionally filter observation types
    if filter_on_observation_types is not None:
        df_y = select_observation_type(df_y, filter_on_observation_types)

    # print('Filter obs type: ', len(df_y))

    # Remove outliers for each observation type
    if remove_outliers:
        df_y = _filter_bbch_outliers(df_y)

    # print('Removed outliers: ', len(df_y))

    # If method is set -> filter data to contain one observation per grid cell
    if aggregation_method is not None:
        df_y = _filter_in_grid(df_y, df_y_loc, method=aggregation_method)

    # print('Filter grid: ', len(df_y))

    # If set -> convert DOY observations to np.datetime64 objects
    if datetime_observations:
        # Create datetime object for the first day in the corresponding year
        years = df_y.index.get_level_values(config.KEY_YEAR).map(lambda x: np.datetime64(str(x), 'Y')).values
        # Create a timedelta based on the DOY observation
        days = df_y[config.KEY_OBSERVATIONS].astype(int).map(lambda x: np.timedelta64(x - 1, 'D')).values
        # Overwrite the DOY observations
        df_y[config.KEY_OBSERVATIONS] = years + days

    return {
        'data': df_y,
        'locations': df_y_loc,
    }


# TODO -- to generalize, the code would require to be able to handle cases where DOYs fall both in dec and jan of subsequent year, so preprocessing to dates would have to be done before
def _filter_bbch_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Quantiles (high and low) to filter
    q = 0.01
    # q = 0.05

    grouped = df.groupby(config.KEY_OBS_TYPE)

    # print([group for _, group in grouped])

    grouped = {k: group[(group[config.KEY_OBSERVATIONS].quantile(q) < group[config.KEY_OBSERVATIONS]) &
                        (group[config.KEY_OBSERVATIONS] < group[config.KEY_OBSERVATIONS].quantile(1 - q))] for k, group in grouped}


    # grouped = {
    #     k: group[np.abs(group[config.KEY_OBSERVATIONS]-group[config.KEY_OBSERVATIONS].mean()) <= (2*group[config.KEY_OBSERVATIONS].std())] for k, group in grouped
    # }

    df = pd.concat(list(grouped.values()))

    return df


def _filter_in_grid(df: pd.DataFrame, df_loc: pd.DataFrame, method: str = None) -> pd.DataFrame:
    assert False, 'Method has been temporarily deprecated and should not be used since it throws away data unnecessarily. Filtering should happen after paining observations based on type'

    df_selection = df.join(df_loc[[config.KEY_LAT, config.KEY_LON]])

    res_lat, res_lon = _AGG_GRID

    df_selection[config.KEY_LAT] = df_selection[config.KEY_LAT].apply(lambda x: round_partial(x, res_lat))
    df_selection[config.KEY_LON] = df_selection[config.KEY_LON].apply(lambda x: round_partial(x, res_lon))

    match method:
        case 'first':

            df_selection.sort_index(inplace=True)
            # Shuffle the dataframe
            df_selection = df_selection.sample(frac=1, random_state=config.RNG_DATA_PREPROCESSING)

            df_selection.reset_index(inplace=True)

            # Aggregate the phenology observations
            df = df_selection.groupby(
                [config.KEY_LAT, config.KEY_LON, config.KEY_OBS_TYPE, config.KEY_YEAR],
                as_index=False,
            ).first()

            df.set_index(list(config.KEYS_INDEX + (config.KEY_OBS_TYPE,)), inplace=True)
            df.drop([config.KEY_LAT, config.KEY_LON], axis=1, inplace=True)

            return df
        case 'mean':

            df_selection.sort_index(inplace=True)
            # Shuffle the dataframe
            df_selection = df_selection.sample(frac=1, random_state=config.RNG_DATA_PREPROCESSING)

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
                    config.KEY_OBSERVATIONS: 'mean',
                    config.KEY_DATA_SOURCE: 'first',
                    config.KEY_LOC_ID: 'first',
                    config.KEY_SPECIES_CODE: 'first',
                    config.KEY_SUBGROUP_CODE: 'first',
                }
            )

            df.reset_index(inplace=True)

            df[config.KEY_OBSERVATIONS] = (df[config.KEY_OBSERVATIONS] + 0.5).astype(int)  # Round to nearest integer

            df.set_index(list(config.KEYS_INDEX + (config.KEY_OBS_TYPE,)), inplace=True)
            df.drop([config.KEY_LAT, config.KEY_LON], axis=1, inplace=True)

            return df

        case 'median':

            df_selection.sort_index(inplace=True)
            # Shuffle the dataframe
            df_selection = df_selection.sample(frac=1, random_state=config.RNG_DATA_PREPROCESSING)

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

            # df[config.KEY_OBSERVATIONS] = (df[config.KEY_OBSERVATIONS] + 0.5).astype(int)  # Round to nearest integer

            df.set_index(list(config.KEYS_INDEX + (config.KEY_OBS_TYPE,)), inplace=True)
            df.drop([config.KEY_LAT, config.KEY_LON], axis=1, inplace=True)

            return df

        case _:
            raise DatasetException(f'Unknown aggregation method "{method}" for phenological observations')


if __name__ == '__main__':

    _result = get_pep725_dataframes()

    # print(_result)

