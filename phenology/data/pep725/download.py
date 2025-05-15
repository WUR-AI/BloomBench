import os
import tarfile
import time

from dataclasses import dataclass

import pandas as pd
import requests

from tqdm import tqdm

from phenology.data.pep725.util import PATH_PEP725, read_df_species_entries, read_df_countries, read_df_species, \
    read_df_species_subgroups, PATH_PEP725_DATA

"""
    Code used to download the PEP725 data from the PEP725 website
    
    Data can be obtained by calling the `get_pep725_data` function, which roughly does the following:
        1. Read which PEP725 is required as specified in the files in the `data.pep725.util.PATH_PEP725_METADATA` folder 
        2. Check which data is already present
        3. Download any data that is missing using the credentials that should be provided
    
"""

# TODO -- preprocess/sanitize read csvs to remove ; in weather station names

# url to PEP725 website
_BASE_URL = 'http://pep725.eu/'
# url to login page
_LOGIN_URL = 'http://pep725.eu/login.php'
# url template for downloading files
_URL_TEMPLATE = 'http://pep725.eu/data_download/download.php?id={country_code}_{species_code:03d}_{subgroup_code:03d}'
# template for naming folders to store the obtained data
_DATA_FOLDER_TEMPLATE = 'PEP725_{country_code}_{species_code:03d}_{subgroup_code:03d}'
# template of how data files are named within the PEP725 data
_FN_TEMPLATE_DATA = 'PEP725_{country_code}_{dfn}.csv'
# template for naming 'raw' download files
_FN_TEMPLATE_DOWNLOAD = 'PEP725_{country_code}_{species_code:03d}_{subgroup_code:03d}.tar.gz'
# bbch code definition filename
_FN_BBCH = 'PEP725_BBCH.csv'
# template of how files containing station data are named
_FN_TEMPLATE_STATIONS = 'PEP725_{country_code}_stations.csv'

# path to the file where PEP725 are stored
_PATH_PEP725_CREDENTIALS = os.path.join(PATH_PEP725, 'credentials.txt')
# path to the folder where raw download files are stored
_PATH_PEP725_DOWNLOADS = os.path.join(PATH_PEP725, 'downloads')
os.makedirs(_PATH_PEP725_DOWNLOADS, exist_ok=True)

# Delay between download requests to the PEP725 website
_DOWNLOAD_DELAY = 0.01  # Seconds


# Define class for conveniently characterizing PEP725 data entries
# An entry corresponds to a single request to the website
# It is uniquely characterized by
# - species code
# - subgroup code (e.g. cultivar)
# - country code
# Other variables are stored for convenience
@dataclass
class PEP725Entry:
    # Key that characterizes the plant species (for readability)
    species_key: str
    # Code that characterizes the plant species as defined by PEP725
    species_code: int
    # Code that characterizes the subgroup as defined by PEP725 (not unique across species)
    subgroup_code: int
    # Code that characterizes the country as defined by PEP725
    country_code: str
    # Species name
    species_name: str
    # Subgroup name
    subgroup_name: str
    # Country name
    country_name: str

    # Define hash function for enabling usage of keys/items in dicts/sets
    def __hash__(self):
        return hash(
            (self.species_key,
             self.species_code,
             self.subgroup_code,
             self.country_code,
             self.species_name,
             self.subgroup_name,
             self.country_name,
             )
        )

    @property
    def download_url(self) -> str:
        """
        Create a url to obtain data corresponding to this entry from the PEP725 website
        :return: the url
        """
        return _URL_TEMPLATE.format(
            country_code=self.country_code,
            species_code=self.species_code,
            subgroup_code=self.subgroup_code,
        )

    @property
    def path_download_file(self) -> str:
        """
        Create a path to the file where data corresponding to this entry will be downloaded
        :return: the path
        """
        fn = _FN_TEMPLATE_DOWNLOAD.format(
            country_code=self.country_code,
            species_code=self.species_code,
            subgroup_code=self.subgroup_code,
        )
        return os.path.join(_PATH_PEP725_DOWNLOADS, fn)

    @property
    def path_data_folder(self) -> str:
        """
        Create a path to the folder where data corresponding to this entry will be stored
        :return: the path
        """
        fn = _DATA_FOLDER_TEMPLATE.format(
            country_code=self.country_code,
            species_code=self.species_code,
            subgroup_code=self.subgroup_code,
        )
        return os.path.join(PATH_PEP725_DATA, self.species_key, fn)

    def get_stations_df(self) -> pd.DataFrame:
        """
        Get a Pandas DataFrame containing all stations corresponding to the country of this entry.
        The dataframe contains the following columns:
        - PEP_ID: the location id as defined by PEP725
        - National_ID: the country id as defined by PEP725
        - LON: the longitude of the station
        - LAT: the latitude of the station
        - ALT: the altitude of the station
        - NAME: the name of the station

        The dataframe is indexed by PEP_ID

        :return: the dataframe
        """
        path = os.path.join(self.path_data_folder,
                            _FN_TEMPLATE_STATIONS.format(country_code=self.country_code),
                            )

        df = pd.read_csv(path, sep=';', index_col=['PEP_ID'])

        df['country_code'] = self.country_code
        return df

    def get_data_df(self, set_index: bool = True) -> pd.DataFrame:
        """
        Get a Pandas DataFrame containing all observations corresponding to this entry
        The dataframe contains the following columns:
        - PEP_ID: the location id as defined by PEP725
        - YEAR: the year of the observation
        - BBCH: the BBCH code of the phenological observation
        - DAY: the day-of-year of the observation
        :param set_index: if True, the dataframe is indexed by PEP_ID and YEAR
        :return: the dataframe
        """
        # Data files do not always have the same name but do follow some structure
        # Downloads corresponding to PEP725 entries always contain the following files:
        #   - PEP725_BBCH.csv
        #   - PEP725_README.txt
        #   - PEP725_{CC}_stations.csv
        #   - PEP725_{CC}_{KEY}.csv            (this file contains the phenology observations)
        # where {CC} is the corresponding country code and {KEY} is a crop species subgroup-specific key.
        # we don't know this key but we can single out the file from the others
        files = os.listdir(self.path_data_folder)
        cc = self.country_code
        files = [fn for fn in files if f'PEP725_{cc}_' in fn and fn != f'PEP725_{cc}_stations.csv']
        assert len(files) == 1  # Sanity check
        fn = files[0]
        # Load the data from the located file path
        df = pd.read_csv(os.path.join(self.path_data_folder, fn), sep=';')
        if set_index:
            df.set_index(['PEP_ID', 'YEAR'], inplace=True)
        return df

    def get_bbch_df(self) -> pd.DataFrame:
        path = os.path.join(self.path_data_folder,
                            _FN_BBCH,
                            )
        df = pd.read_csv(path, sep=';')
        df.set_index('bbch', inplace=True)

        # Some BBCH code definitions are missing
        # Add required but missing entries (obtained from http://pep725.eu/pep725_phase.php)
        df.loc[50] = 'Flower buds present, still enclosed by leaves (oilseed rape)'

        df.sort_index(inplace=True)
        return df


def get_pep725_data(verbose: bool = True,
                    force_download: bool = False,  # TODO -- change to [enabled, disabled, forced]
                    ) -> dict:  # TODO -- option to select subset of data
    """

    Obtain the required PEP725 data as listed in the files contained in the folder specified in `data.pep725.util.PATH_PEP725_METADATA`
    :param verbose: boolean flag indicating whether a progress bar should be printed
    :param force_download: If set, forces download of data even if it already exists
    :return: a dict containing
     - 'data': a pandas DataFrame containing PEP725 phenological observations
        The dataframe is indexed by:
            'PEP_ID', 'YEAR', 'species_code', 'subgroup_code', 'BBCH'
        and contains a column 'DAY' with the day-of-year of the observation

     - 'entries':
        a dict showing which of the required data entries are present or might still be missing

     - 'stations': a pandas DataFrame containing info of the relevant weather stations
        The dataframe is indexed by PEP_ID identifying the station
        The dataframe contains the following columns:
        National_ID - The ID of the country in which the station is located
        LON - Longitude of the station
        LAT - Latitude of the station
        ALT - Altitude of the station
        NAME - Name of the station
        country_code - Country code of the country in which the station is located
    """

    # Obtain an overview of PEP725 data
    df_entries = read_df_species_entries()
    df_countries = read_df_countries()
    df_species = read_df_species()
    df_species_subgroups = read_df_species_subgroups()

    # List all entries
    entries = {
        PEP725Entry(species_key=df_species.loc[species_code].key,
                    species_code=species_code,
                    subgroup_code=subgroup_code,
                    country_code=country_code,
                    species_name=df_species.loc[species_code].species,
                    subgroup_name=df_species_subgroups.loc[species_code, subgroup_code].subgroup_name,
                    country_name=df_countries.loc[country_code].country_name,
                    ) for _, species_code, subgroup_code, country_code in df_entries.itertuples()
    }

    """
        Check which entries are missing data or raw download files
    """

    # Keep track of which entries for which download files should be extracted
    entries_to_extract = set()
    # Keep track for which entries no data could be obtained
    entries_failed = set()
    # Keep track of which entries should be downloaded
    if force_download:  # If set -> download all files
        entries_to_download = entries
    else:  # If not set -> check for missing data
        entries_to_download = set()

        # Check if any data is missing either raw download files or data
        entries_missing = _check_entries_missing_data(entries,
                                                      verbose=verbose,
                                                      )

        # For entries that miss data
        for entry in entries_missing['data']:
            # If raw download files are present -> no download required
            if entry not in entries_missing['download']:
                entries_to_extract.add(entry)
            else:  # If not present -> add to entries for which files should be downloaded
                entries_to_download.add(entry)

    """
        Download required files
    """

    # Download the required PEP725 data
    result_download = _download_pep725_entries(entries_to_download,
                                               verbose=verbose,
                                               )
    # Successful downloads should be extracted
    entries_to_extract = entries_to_extract.union(result_download['successful'])
    # Unsuccessful downloads result in the data being unobtainable
    entries_failed = entries_failed.union(result_download['failed'])

    """
        Extract required files
    """
    # Extract raw download files to the right data folders
    result_extract = _extract_pep725_entries(entries_to_extract,
                                             verbose=verbose,
                                             )

    # Return a dict containing
    return {
        # A dataframe containing all observations
        'data': _create_data_df(entries - entries_failed),

        # Categorization of entries
        'entries': {
            # A set of entries that are included in the data
            'included': entries - entries_failed,
            # A set of entries for which data was downloaded
            'downloads': result_download['successful'],
            # A set of entries for which data was extracted
            'extracted': entries_to_extract,
            # A set of entries for which no data could be obtained
            'failed': entries_failed,
            # The total set of entries that was requested
            'requested': entries,
        },

        # A dataframe containing all stations corresponding to the observations
        'stations': _create_stations_df(entries),
    }


def _check_entries_missing_data(entries: set,
                                verbose: bool = True,
                                ) -> dict:

    entries_missing_data = []
    entries_missing_download = []

    if verbose:
        iterable = tqdm(entries,
                        total=len(entries),
                        desc='Checking for missing PEP725 data',
                        )
    else:
        iterable = entries

    for entry in iterable:

        if not os.path.exists(entry.path_data_folder):
            entries_missing_data.append(entry)

        if not os.path.exists(entry.path_download_file):
            entries_missing_download.append(entry)

        if verbose:
            iterable.set_postfix({
                'species': f'{entry.species_code} ({entry.species_name})',
                'subgroup': f'{entry.subgroup_code} ({entry.subgroup_name})',
                'country': f'{entry.country_code} ({entry.country_name})',
                'n_missing_data': f'{len(entries_missing_data)}/{len(entries)}',
                'n_missing_download': f'{len(entries_missing_download)}/{len(entries)}',
            })

    return {
        'data': entries_missing_data,
        'download': entries_missing_download,
    }


def _download_pep725_entries(entries: set,
                             verbose: bool = True,
                             ) -> dict:

    entries_success = set()
    entries_failed = set()

    if len(entries) > 0:

        session = requests.Session()

        username, password = _read_credentials_file()

        response = session.post(_LOGIN_URL,
                                data={
                                    'email': username,
                                    'pwd': password,
                                    'submit': 'Login',
                                })

        if verbose:
            iterable = tqdm(entries,
                            total=len(entries),
                            desc='Downloading PEP725 data',
                            )
        else:
            iterable = entries

        num_failed = 0
        for entry in iterable:

            response = session.get(entry.download_url,
                                   cookies=session.cookies.get_dict(),
                                   )

            success = response.headers['Content-Type'] == 'application/x-gzip'

            if success:
                path_download = entry.path_download_file
                os.makedirs(os.path.dirname(path_download), exist_ok=True)
                with open(path_download, 'wb') as f:
                    f.write(response.content)
                entries_success.add(entry)
            else:
                num_failed += 1
                entries_failed.add(entry)

            if verbose:
                iterable.set_postfix({
                    # 'species': f'{entry.species_code} ({entry.species_name})',
                    # 'subgroup': f'{entry.subgroup_code} ({entry.subgroup_name})',
                    # 'country': f'{entry.country_code} ({entry.country_name})',
                    # 'success': f'{success}',
                    'n_failed': f'{num_failed}/{len(entries)}',
                })

            time.sleep(_DOWNLOAD_DELAY)

    return {
        'successful': entries_success,
        'failed': entries_failed,
    }


def _extract_pep725_entries(entries: set,
                            verbose: bool = True,
                            ) -> dict:

    if len(entries) > 0:

        if verbose:
            iterable = tqdm(entries,
                            total=len(entries),
                            desc='Extracting PEP725 data',
                            )
        else:
            iterable = entries

        for entry in iterable:

            with tarfile.open(entry.path_download_file, 'r:gz') as f:
                f.extractall(path=entry.path_data_folder)

            path_stations = os.path.join(entry.path_data_folder, _FN_TEMPLATE_STATIONS.format(country_code=entry.country_code))
            _sanitize_stations_csv(path_stations)

            if verbose:
                iterable.set_postfix({
                    'species': f'{entry.species_code} ({entry.species_name})',
                    'subgroup': f'{entry.subgroup_code} ({entry.subgroup_name})',
                    'country': f'{entry.country_code} ({entry.country_name})',
                })

    return {
        # TODO
    }


def _sanitize_stations_csv(path: str):
    """
    Some station names in PEP725 files contain semicolons. Since this data is stored in semicolon-separated csv files
    this prevents us from reading the file. This function checks and removes any semicolons in this column
    i.e.
    Reads a ;-separated csv file and ensures that each row contains exactly 6 semicolons.
    If there are more than 6, replaces the last few semicolons with spaces until each line has 6.
    """
    with open(path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    processed_lines = []
    for line in lines:
        # Split the line into parts using ';' as delimiter
        parts = line.strip().split(';')

        # If there are more than 6 semicolons, replace the excess with spaces
        if len(parts) > 6:
            # Take the first 6 parts and join them with ';'
            processed_line = ';'.join(parts[:6])

            remaining_parts = ' '.join(parts[6:])
            processed_line += remaining_parts
        else:
            processed_line = line.strip()

        processed_lines.append(processed_line + '\n')

    # Write the processed lines to the output file
    with open(path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(processed_lines)


def _read_credentials_file() -> tuple:
    assert os.path.exists(_PATH_PEP725_CREDENTIALS), \
        f'PEP725 credentials file ({_PATH_PEP725_CREDENTIALS}) does not exist'

    with open(_PATH_PEP725_CREDENTIALS, 'r') as f:
        tokens = f.read().split(' ')
        username = tokens[0]
        password = ' '.join(tokens[1:])
    return username, password


def _create_stations_df(entries: set) -> pd.DataFrame:
    """
    Create a dataframe containing all stations occurring in the PEP725 data
    The dataframe is indexed by PEP_ID identifying the station
    The dataframe contains the following columns:
        National_ID - The ID of the country in which the station is located
        LON - Longitude of the station
        LAT - Latitude of the station
        ALT - Altitude of the station
        NAME - Name of the station
        country_code - Country code of the country in which the station is located
    :param entries: a set of PEP725 entries
    :return: a dataframe containing all stations occurring in the PEP725 data
    """
    # Each entry has a csv containing all stations for the respective country
    # For each country, we pick one entry and read this csv
    picked_entries = dict()
    for entry in entries:
        picked_entries[entry.country_code] = entry
    # Read the csv for each country occurring in the entries
    dfs = [e.get_stations_df() for e in picked_entries.values()]
    # Merge all data into one pandas dataframe
    df = pd.concat(dfs)
    return df


def _create_data_df(entries: set) -> pd.DataFrame:
    """
    Create a dataframe containing all observations corresponding to the PEP725 data entries
    :param entries:
    :return:
    """

    dfs = []
    for entry in entries:
        df = entry.get_data_df(set_index=False)
        df['species_code'] = entry.species_code
        df['subgroup_code'] = entry.subgroup_code
        df.set_index(['PEP_ID', 'YEAR', 'species_code', 'subgroup_code', 'BBCH'], inplace=True)
        dfs.append(df)

    df = pd.concat(dfs)
    return df


if __name__ == '__main__':

    result = get_pep725_data()

    # df = _create_stations_df(result['entries'])
    df = result['data']

    print(df)

    # print(df[['PEP_ID']].nunique())

    # 333 wheat
    # 330 barley
    # 332 rye

    df = df.xs(332, level='species_code')

    for bbch, df_bbch in df.groupby('BBCH'):
        print(bbch, len(df_bbch))

