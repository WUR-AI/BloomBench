import os

import pandas as pd

from phenology.config import PATH_DATA

SRC_KEY = 'pep725'

PATH_PEP725 = os.path.join(PATH_DATA, 'pep725')
os.makedirs(PATH_PEP725, exist_ok=True)

PATH_PEP725_DATA = os.path.join(PATH_PEP725, 'data')
os.makedirs(PATH_PEP725_DATA, exist_ok=True)

PATH_PEP725_METADATA = os.path.join(PATH_PEP725, 'metadata')
os.makedirs(PATH_PEP725_METADATA, exist_ok=True)


def read_df_species() -> pd.DataFrame:
    return pd.read_csv(os.path.join(PATH_PEP725_METADATA, 'species.csv'),
                       sep=';',
                       index_col='species_code',
                       )


def read_df_countries() -> pd.DataFrame:
    return pd.read_csv(os.path.join(PATH_PEP725_METADATA, 'countries.csv'),
                       sep=';',
                       index_col='country_code',
                       )


def read_df_species_subgroups() -> pd.DataFrame:
    return pd.read_csv(os.path.join(PATH_PEP725_METADATA, 'species_subgroups.csv'),
                       sep=';',
                       index_col=('species_code', 'subgroup_code'),
                       )


def read_df_species_entries() -> pd.DataFrame:
    return pd.read_csv(os.path.join(PATH_PEP725_METADATA, 'species_entries.csv'),
                       sep=';',
                       )


if __name__ == '__main__':

    print(read_df_species())
    print(read_df_countries())
    print(read_df_species_subgroups())
    print(read_df_species_entries())
