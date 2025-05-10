
import pandas as pd

from phenology import config


def select_species(df: pd.DataFrame, species: tuple) -> pd.DataFrame:
    assert isinstance(species, tuple) or isinstance(species, list)

    if isinstance(species, list):
        if len(species) == 0:
            return empty_df_like(df)
        else:
            return pd.concat([select_species(df, s) for s in species], axis=0)

    src, spc, sub = species
    try:
        df = df.xs((src, spc, sub),
                   level=[
                       config.KEY_DATA_SOURCE,
                       config.KEY_SPECIES_CODE,
                       config.KEY_SUBGROUP_CODE,
                   ],
                   drop_level=False,
                   )
        return df
    except KeyError:
        return empty_df_like(df)


def select_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    assert isinstance(year, int) or isinstance(year, list)

    if isinstance(year, list):
        if len(year) == 0:
            return empty_df_like(df)
        else:
            return pd.concat([select_year(df, y) for y in year], axis=0)

    try:
        df = df.xs(year,
                   level=config.KEY_YEAR,
                   drop_level=False,
                   )
        return df
    except KeyError:
        return empty_df_like(df)


def select_location(df: pd.DataFrame, location: tuple) -> pd.DataFrame:
    assert isinstance(location, tuple) or isinstance(location, list)

    if isinstance(location, list):
        if len(location) == 0:
            return empty_df_like(df)
        else:
            return pd.concat([select_location(df, l) for l in location], axis=0)

    src, loc_id = location
    try:
        df = df.xs((src, loc_id),
                   level=[
                       config.KEY_DATA_SOURCE,
                       config.KEY_LOC_ID,
                   ],
                   drop_level=False,
                   )
        return df
    except KeyError:
        return empty_df_like(df)


def select_observation_type(df: pd.DataFrame, obs_type: tuple) -> pd.DataFrame:

    if isinstance(obs_type, list):
        if len(obs_type) == 0:
            return empty_df_like(df)
        else:
            return pd.concat([select_observation_type(df, t) for t in obs_type], axis=0)

    try:
        df = df.xs(obs_type,
                   level=config.KEY_OBS_TYPE,
                   drop_level=False,
                   )
        return df
    except KeyError:
        return empty_df_like(df)


def empty_df_like(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=df.columns).reindex_like(df).dropna().astype(df.dtypes)
