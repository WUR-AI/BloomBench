import numpy as np
import pandas as pd

from phenology import config
from phenology.data.gmu_cherry.bloom_doy import get_df_japan, get_df_switzerland, get_df_south_korea
from phenology.data.gmu_cherry.regions_data import LOCATION_VARIETY_JAPAN


def get_gmu_cherry_dataset_japan(remove_outliers: bool = True,
                                 datetime_observations: bool = True,
                                 ) -> dict:

    src = config.KEY_GMU_CHERRY

    df_data = get_df_japan()

    df_data.reset_index(inplace=True)

    df_data['location_name'] = df_data['location']
    df_data['location'] = df_data['location'].map(lambda x: x.replace('/', '__'))

    df_y_loc = df_data[['location', 'lat', 'long', 'alt']].copy()
    df_y_loc[config.KEY_DATA_SOURCE] = src
    df_y_loc[config.KEY_COUNTRY_CODE] = 'JP'
    df_y_loc[config.KEY_LOC_NAME] = df_data['location_name']
    df_y_loc.rename(
        columns={
            'location': config.KEY_LOC_ID,
            'lat': config.KEY_LAT,
            'long': config.KEY_LON,
            'alt': config.KEY_ALT,
        },
        inplace=True,
    )
    df_y_loc.set_index([config.KEY_DATA_SOURCE, config.KEY_LOC_ID], inplace=True)
    df_y_loc.drop_duplicates(keep='first', inplace=True)

    bloom_obs_type = 'gmu_0'  # TODO -- proper name

    df_y = df_data[['location', 'year', 'bloom_doy']].copy()
    df_y[config.KEY_DATA_SOURCE] = src
    df_y[config.KEY_OBS_TYPE] = bloom_obs_type

    loc_species_map = LOCATION_VARIETY_JAPAN
    loc_species_map = {k.replace('/', '__'): v for k, v in loc_species_map.items()}

    # Filter on locations that have species information
    df_y = df_y[df_y['location'].isin(loc_species_map.keys())]
    species = df_y['location'].map(loc_species_map)

    df_y[config.KEY_SPECIES_CODE] = species
    df_y[config.KEY_SUBGROUP_CODE] = 0  # Unknown
    df_y.rename(
        columns={
            'location': config.KEY_LOC_ID,
            'year': config.KEY_YEAR,
            'bloom_doy': config.KEY_OBSERVATIONS,
        },
        inplace=True,
    )
    df_y.set_index([
        config.KEY_DATA_SOURCE,
        config.KEY_LOC_ID,
        config.KEY_YEAR,
        config.KEY_SPECIES_CODE,
        config.KEY_SUBGROUP_CODE,
        config.KEY_OBS_TYPE,
    ], inplace=True)

    # Remove outliers for each observation type
    if remove_outliers:
        df_y = _filter_bbch_outliers(df_y)

    # If set -> convert DOY observations to np.datetime64 objects
    if datetime_observations:
        # Create datetime object for the first day in the corresponding year
        years = df_y.index.get_level_values(config.KEY_YEAR).map(lambda x: np.datetime64(str(x), 'Y')).values.astype('datetime64[D]')
        # Create a timedelta based on the DOY observation
        days = (df_y[config.KEY_OBSERVATIONS].values - 1).astype('timedelta64[D]')
        # Overwrite the DOY observations
        df_y[config.KEY_OBSERVATIONS] = years + days

    return {
        'data': df_y,
        'locations': df_y_loc,
    }


def get_gmu_cherry_dataset_switzerland(remove_outliers: bool = True,
                                       datetime_observations: bool = True,
                                       ) -> dict:

    src = config.KEY_GMU_CHERRY

    df_data = get_df_switzerland()

    df_data.reset_index(inplace=True)

    df_data['location_name'] = df_data['location']
    df_data['location'] = df_data['location'].map(lambda x: x.replace('/', '__'))

    df_y_loc = df_data[['location', 'lat', 'long', 'alt']].copy()
    df_y_loc[config.KEY_DATA_SOURCE] = src
    df_y_loc[config.KEY_COUNTRY_CODE] = 'SW'
    df_y_loc[config.KEY_LOC_NAME] = df_data['location_name']
    df_y_loc.rename(
        columns={
            'location': config.KEY_LOC_ID,
            'lat': config.KEY_LAT,
            'long': config.KEY_LON,
            'alt': config.KEY_ALT,
        },
        inplace=True,
    )
    df_y_loc.set_index([config.KEY_DATA_SOURCE, config.KEY_LOC_ID], inplace=True)
    df_y_loc.drop_duplicates(keep='first', inplace=True)

    bloom_obs_type = 'gmu_1'  # TODO

    df_y = df_data[['location', 'year', 'bloom_doy']].copy()
    df_y[config.KEY_DATA_SOURCE] = src
    df_y[config.KEY_OBS_TYPE] = bloom_obs_type
    df_y[config.KEY_SPECIES_CODE] = 5
    df_y[config.KEY_SUBGROUP_CODE] = 0
    df_y.rename(
        columns={
            'location': config.KEY_LOC_ID,
            'year': config.KEY_YEAR,
            'bloom_doy': config.KEY_OBSERVATIONS,
        },
        inplace=True,
    )
    df_y.set_index([
        config.KEY_DATA_SOURCE,
        config.KEY_LOC_ID,
        config.KEY_YEAR,
        config.KEY_SPECIES_CODE,
        config.KEY_SUBGROUP_CODE,
        config.KEY_OBS_TYPE,
    ], inplace=True)

    # Remove outliers for each observation type
    if remove_outliers:
        df_y = _filter_bbch_outliers(df_y)

    # If set -> convert DOY observations to np.datetime64 objects
    if datetime_observations:
        # Create datetime object for the first day in the corresponding year
        years = df_y.index.get_level_values(config.KEY_YEAR).map(lambda x: np.datetime64(str(x), 'Y')).values.astype('datetime64[D]')
        # Create a timedelta based on the DOY observation
        days = (df_y[config.KEY_OBSERVATIONS].values - 1).astype('timedelta64[D]')
        # Overwrite the DOY observations
        df_y[config.KEY_OBSERVATIONS] = years + days

        # Overwrite the DOY observations
        df_y[config.KEY_OBSERVATIONS] = years + days

    return {
        'data': df_y,
        'locations': df_y_loc,
    }


def get_gmu_cherry_dataset_south_korea(remove_outliers: bool = True,
                                       datetime_observations: bool = True,
                                       ) -> dict:

    src = config.KEY_GMU_CHERRY

    df_data = get_df_south_korea()

    df_data.reset_index(inplace=True)

    df_data['location_name'] = df_data['location']
    df_data['location'] = df_data['location'].map(lambda x: x.replace('/', '__'))

    df_y_loc = df_data[['location', 'lat', 'long', 'alt']].copy()
    df_y_loc[config.KEY_DATA_SOURCE] = src
    df_y_loc[config.KEY_COUNTRY_CODE] = 'KR'
    df_y_loc[config.KEY_LOC_NAME] = df_data['location_name']
    df_y_loc.rename(
        columns={
            'location': config.KEY_LOC_ID,
            'lat': config.KEY_LAT,
            'long': config.KEY_LON,
            'alt': config.KEY_ALT,
        },
        inplace=True,
    )
    df_y_loc.set_index([config.KEY_DATA_SOURCE, config.KEY_LOC_ID], inplace=True)
    df_y_loc.drop_duplicates(keep='first', inplace=True)

    bloom_obs_type = 'gmu_2'

    df_y = df_data[['location', 'year', 'bloom_doy']].copy()
    df_y[config.KEY_DATA_SOURCE] = src
    df_y[config.KEY_OBS_TYPE] = bloom_obs_type
    df_y[config.KEY_SPECIES_CODE] = 0
    df_y[config.KEY_SUBGROUP_CODE] = 0
    df_y.rename(
        columns={
            'location': config.KEY_LOC_ID,
            'year': config.KEY_YEAR,
            'bloom_doy': config.KEY_OBSERVATIONS,
        },
        inplace=True,
    )
    df_y.set_index([
        config.KEY_DATA_SOURCE,
        config.KEY_LOC_ID,
        config.KEY_YEAR,
        config.KEY_SPECIES_CODE,
        config.KEY_SUBGROUP_CODE,
        config.KEY_OBS_TYPE,
    ], inplace=True)

    # Remove outliers for each observation type
    if remove_outliers:
        df_y = _filter_bbch_outliers(df_y)

    # If set -> convert DOY observations to np.datetime64 objects
    if datetime_observations:
        # Create datetime object for the first day in the corresponding year
        years = df_y.index.get_level_values(config.KEY_YEAR).map(lambda x: np.datetime64(str(x), 'Y')).values.astype('datetime64[D]')
        # Create a timedelta based on the DOY observation
        days = (df_y[config.KEY_OBSERVATIONS].values - 1).astype('timedelta64[D]')
        # Overwrite the DOY observations
        df_y[config.KEY_OBSERVATIONS] = years + days

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


if __name__ == '__main__':
    dfs = get_gmu_cherry_dataset_japan(datetime_observations=True)

    print(dfs['data'])
    # print(dfs['locations'])

    get_gmu_cherry_dataset_switzerland(datetime_observations=False)
    get_gmu_cherry_dataset_south_korea(datetime_observations=False)
