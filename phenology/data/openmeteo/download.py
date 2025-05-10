import os
import time

import hashlib

from dataclasses import dataclass

import pandas as pd
from openmeteo_requests.Client import OpenMeteoRequestsError
from tqdm import tqdm
import tables  # Required for storing data

import openmeteo_requests
import requests_cache
from retry_requests import retry

import phenology.config
from phenology.data.openmeteo import PATH_OPENMETEO, PATH_OPENMETEO_DATA

"""

    Code for downloading and accessing OpenMeteo data.

    DESCRIPTION:

    OpenMeteo data can be requested per entry

    Data entries are characterized by:
        - Data source key
        - Location key as defined by the data source
        - Year 

    Each request will obtain hourly temperature data from OpenMeteo for that year in the specified location.
    Location coordinates are also provided when creating the entries

    Given these entries, checks are made whether the data is already present or should be downloaded.

    Data is downloaded from the OpenMeteo API and stored in a HDF5 store.
    The OpenMeteoStore class is used to interact with the HDF5 store.

"""

_STORE_NAME_TEMPLATE = "openmeteo_data_store_{step}_{data_key}.h5"
_STORE_PATH_TEMPLATE = os.path.join(PATH_OPENMETEO_DATA, _STORE_NAME_TEMPLATE)

# Template of keys that are used to store data
_STORE_KEY_TEMPLATE = "/openmeteo/{step}/{data_key}/group_{group_key}/src_{src_key}/loc_{loc_id}/year_{year}"

# Define/create folders for storing openmeteo data
# _PATH_DATA_STORE = os.path.join(PATH_OPENMETEO_DATA, 'openmeteo_data_store.h5')

# Path of where to look for the openmeteo API key
_PATH_API_KEY = os.path.join(PATH_OPENMETEO, 'openmeteo_api_key.txt')

# Delay between download requests
_DOWNLOAD_DELAY = 0.01  # Seconds

# Data compression settings (DON'T CHANGE -- OR RISK CORRUPTING THE EXISTING DATA STORE)
_STORE_COMP_LIB = 'zlib'
_STORE_COMP_LVL = 5
_STORE_FORMAT = 'table'


@dataclass
class OpenMeteoEntry:
    """
    Class for conveniently characterizing openmeteo data entries

    Entries should be uniquely characterized by (step, data_key, src_key, loc_id, year), where
      - step is the temporal resolution of the data (e.g. hourly, daily, etc.)
      - data_key is the key used to identify the data type (see openmeteo docs)
      - src_key is the key used to identify the phenology data source
      - loc_id is the location key used to identify the location
      - year identifies the season that is considered
    The remaining variables are stored for ease of access

    """

    # Temporal resolution of the data
    step: str
    # Data type key
    data_key: str
    # Identifier for the label data source
    # In combination with the loc_id (within that data source) it uniquely characterizes each location
    src_key: str
    # Location ID
    loc_id: int
    # Location name
    loc_name: str
    # Latitude
    lat: float
    # Longitude
    lon: float
    # Data is stored/obtained per year/season
    year: int
    # # Country code  (for convenience)
    # country_code: str

    # Define hash function for enabling usage of keys/items in dicts/sets
    def __hash__(self):
        return hash(
            (
                self.step,
                self.data_key,
                self.src_key,
                self.loc_id,
                self.year,
             )
        )


class OpenMeteoStores:
    """
        Class for accessing OpenMeteo data through the HDF5 Store

        Abstracts away some of the required steps, e.g. converting data items to their corresponding HDF5 store keys
    """
    def __init__(self, step_key_pairs: list):

        self._stores = {
            (step, key): OpenMeteoStores._load_store(step, key) for step, key in step_key_pairs
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getitem__(self, item: tuple) -> pd.DataFrame:
        if isinstance(item, OpenMeteoEntry):
            item = OpenMeteoStores._entry_to_tuple(item)

        key = OpenMeteoStores._get_store_key(item)
        store = self._get_store(item)
        o = store.get(key=key)
        assert isinstance(o, pd.DataFrame)
        return o

    def __setitem__(self, item: tuple, df: pd.DataFrame):  # TODO -- check / in key
        if isinstance(item, OpenMeteoEntry):
            item = OpenMeteoStores._entry_to_tuple(item)

        key = OpenMeteoStores._get_store_key(item)
        store = self._get_store(item)

        df.to_hdf(store,
                  key=key,
                  format=_STORE_FORMAT,
                  complib=_STORE_COMP_LIB,
                  complevel=_STORE_COMP_LVL,
                  )

    def __contains__(self, item: tuple) -> bool:
        if isinstance(item, OpenMeteoEntry):
            item = OpenMeteoStores._entry_to_tuple(item)

        key = OpenMeteoStores._get_store_key(item)
        store = self._get_store(item)
        return key in store

    def close(self):
        for store in self._stores.values():
            store.close()

    def _get_store(self, item) -> pd.HDFStore:
        step, data_key, src_key, loc_id, year = item
        return self._stores[step, data_key]

    @staticmethod
    def _entry_to_tuple(entry: OpenMeteoEntry) -> tuple:
        return entry.step, entry.data_key, entry.src_key, entry.loc_id, entry.year

    @staticmethod
    def _get_store_path(step: str, key: str) -> str:
        return _STORE_PATH_TEMPLATE.format(step=step, data_key=key)

    @staticmethod
    def _load_store(step: str, key: str) -> pd.HDFStore:
        return pd.HDFStore(OpenMeteoStores._get_store_path(step, key),
                           complib=_STORE_COMP_LIB,
                           complevel=_STORE_COMP_LVL,
                           )

    @staticmethod
    def _get_store_key(item: tuple) -> str:
        """
        Create the key used to store the data entry in the HDFStore object
        :param item: the data entry
        :return: a store key (str)
        """
        step, data_key, src_key, loc_id, year = item
        group_key = OpenMeteoStores._assign_group_key(item)
        key = _STORE_KEY_TEMPLATE.format(
            step=step,
            data_key=data_key,
            group_key=group_key,
            src_key=src_key,
            loc_id=loc_id,
            year=year,
        )
        return key

    @staticmethod
    def _assign_group_key(item: tuple) -> str:
        """
        Assign group key to an openmeteo store entry.
        Entries are grouped since pytables strongly recommends group width/sizes under ~16000
        Assigns keys s.t. entries are distributed evenly across groups.
        Keys are determined by (src_key, loc_id, year)
        :param item: tuple entry
        :return: a group key (str)
        """
        step, data_key, src_key, loc_id, year = item
        # Generate a key by hashing the tuple that characterizes the entry
        # Use hashlib since the default python hash is not deterministic across runs
        s = f'{src_key}-{loc_id}-{year}'
        h = hashlib.sha1(bytearray(s, encoding='utf-8'), usedforsecurity=False).hexdigest()
        # Take only first three hex characters (for max 16^3=4096 groups)
        key = h[:3]

        return key

    def _transfer_store(self, name: str = 'openmeteo_data_store.h5'):
        path = os.path.join(PATH_OPENMETEO_DATA, name)
        with pd.HDFStore(path, complib=_STORE_COMP_LIB, complevel=_STORE_COMP_LVL) as store:
            store_keys = store.keys()
            for key in tqdm(store_keys):
                tokens = key.split('/')

                # Select tokens
                step = tokens[2]
                data_key = tokens[3]
                # src_key = tokens[5]
                # loc_id = tokens[6]
                # year = tokens[7]
                #
                # # Remove first part of token
                # src_key = '_'.join(src_key.split('_')[1:])
                # loc_id = '_'.join(loc_id.split('_')[1:])
                # year = '_'.join(year.split('_')[1:])

                df = store.get(key=key)
                assert isinstance(df, pd.DataFrame)

                if (step, data_key) not in self._stores.keys():
                    print('Store is not completely transferred! ', (step, data_key))
                    continue

                # print(step, data_key)

                df.to_hdf(self._stores[step, data_key],
                          key=key,
                          format=_STORE_FORMAT,
                          complib=_STORE_COMP_LIB,
                          complevel=_STORE_COMP_LVL,
                          )


"""
    ####################################################################################################################
    # Functions for downloading data from OpenMeteo API                                                                #
    ####################################################################################################################
"""


DOWNLOAD_MODES = [
    None,  # Default -- download all missing data
    'forced',  # Download all data, regardless of whether its missing
    'skip',  # Skip downloading data
]


def get_openmeteo_data(entries: set,
                       verbose: bool = True,
                       download_mode: str = None,
                       ) -> dict:  # TODO -- option to retry download if it fails
    """
    Function for downloading the specified OpenMeteo data.
    :param entries: entries corresponding to the OpenMeteo data.
    :param verbose: flag for verbose output.
    :param download_mode:
    :return:
    """
    assert download_mode in DOWNLOAD_MODES, f'Unrecognized download mode: {download_mode}'

    if len(entries) != 0:

        match download_mode:
            case None:

                step_key_pairs = list(set([(e.step, e.data_key) for e in entries]))

                with OpenMeteoStores(step_key_pairs) as stores:

                    entries_missing = _check_entries_missing_data(entries,
                                                                  stores,
                                                                  verbose=verbose,
                                                                  )

                    result_download = _download_openmeteo_entries(entries_missing,
                                                                  stores,
                                                                  verbose=verbose,
                                                                  )

                    # entries_failed = entries_failed.union(result_download['failed'])

            case 'forced':

                step_key_pairs = list(set([(e.step, e.data_key) for e in entries]))

                with OpenMeteoStores(step_key_pairs) as stores:

                    result_download = _download_openmeteo_entries(entries,
                                                                  stores,
                                                                  verbose=verbose,
                                                                  )

                    # entries_failed = entries_failed.union(result_download['failed'])

            case 'skip':
                pass
            case _:
                raise OpenMeteoDownloadException(f'Unrecognized download mode: {download_mode}')

    return {
        # 'store': OpenMeteoStore(),
    }


def _check_entries_missing_data(entries: set,
                                stores: OpenMeteoStores,
                                verbose: bool = True,
                                ) -> set:
    """
    Helper function
    Iterates through all entries to check if their corresponding data is present
    :param entries: the entries to check
    :param stores: the store to check for data
    :param verbose:
    :return:
    """
    if verbose:
        iterable = tqdm(entries,
                        total=len(entries),
                        desc='Checking for missing meteo data',
                        )
    else:
        iterable = entries

    entries_missing_data = set()
    for entry in iterable:
        if entry not in stores:
            entries_missing_data.add(entry)

        if verbose:
            iterable.set_postfix({
                'step': f'{entry.step}',
                'data_key': f'{entry.data_key}',
                'location_id': f'({entry.src_key}, {entry.loc_id})',
                # 'coordinates': f'({entry.lat} °N, {entry.lon} °E)',
                'year': f'{entry.year}',
                # 'location': f'{entry.loc_name} ({entry.country_code})',
                'n_missing_data': f'{len(entries_missing_data)}/{len(entries)}',
            })

    return entries_missing_data


def _download_openmeteo_entries(entries: set,
                                stores: OpenMeteoStores,
                                verbose: bool = True,
                                ) -> dict:  # TODO -- option for retrying download until complete
    """
    Helper function
    Downloads the specified OpenMeteo data
    :param entries: the entries to download
    :param stores: where to store the data
    :param verbose: flag for verbose output.
    :return:
    """

    # Useful documentation:
    # https://open-meteo.com/en/docs
    # https://pypi.org/project/openmeteo-py/
    # https://pypi.org/project/openmeteo-sdk/

    if len(entries) > 0:

        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        # Read the API key if present -- returns None otherwise
        api_key = _get_api_key()
        # Set boolean that indicates whether an API key should be used
        commercial = api_key is not None

        if api_key is None:
            print('No OpenMeteo API key found. Proceeding with the non-commercial API limitations')
        else:
            print('OpenMeteo API key found. Using commercial API')

        # Create data source url based on whether an API key is present
        url = _get_base_url(commercial=commercial)

        # Create an iterator for iterating through all data entries
        if verbose:
            iterable = tqdm(entries,
                            total=len(entries),
                            desc='Downloading openmeteo data',
                            )
        else:
            iterable = entries

        # Keep a counter of entries that failed to download
        num_failed = 0
        # Try downloading each entry
        for entry in iterable:

            # Get the temporal resolution of this entry
            step = entry.step

            parameters = {
                "latitude": entry.lat,
                "longitude": entry.lon,
                # "hourly": "temperature_2m",
                "start_date": f"{entry.year}-01-01",
                "end_date": f"{entry.year}-12-31",
                # "models": "cerra",
                "models": "era5",
            }

            if step == 'hourly':
                parameters['hourly'] = entry.data_key
            if step == 'daily':
                parameters['daily'] = entry.data_key

            if commercial:
                parameters['apikey'] = api_key

            try:
                # Response documentation: https://github.com/open-meteo/sdk
                responses = openmeteo.weather_api(url, params=parameters)

                # Process first location. Add a for-loop for multiple locations or weather models
                response = responses[0]
                # print(response)
                # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
                # print(f"Elevation {response.Elevation()} m asl")
                # print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
                # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

                # # Process hourly data. The order of variables needs to be the same as requested.
                # hourly = response.Hourly()
                # hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

                match step:
                    case 'hourly':
                        data = response.Hourly()
                    case 'daily':
                        data = response.Daily()
                    case _:
                        raise OpenMeteoDownloadException(f'Unsupported value for "step": {step}')

                # data = data.Variables(0).ValuesAsNumpy()

                # print(response.Timezone())

                data = {
                    "date": pd.date_range(
                                start=pd.to_datetime(
                                    data.Time(),
                                    unit="s",
                                    utc=True,
                                ),
                                end=pd.to_datetime(
                                    data.TimeEnd(),
                                    unit="s",
                                    utc=True,
                                ),
                                freq=pd.Timedelta(seconds=data.Interval()),
                                inclusive="left",
                    ),
                    entry.data_key: data.Variables(0).ValuesAsNumpy(),
                }

                # hourly_data['date'].tz_convert(response.Timezone())  # This doesn't work -- response.Timezone() is often None. UTC is sufficient

                df = pd.DataFrame(data=data)
                # hourly_dataframe.set_index('date', inplace=True)

                # df.to_hdf(store,
                #           key=entry.get_store_key(),
                #           format=_STORE_FORMAT,
                #           complib=_STORE_COMP_LIB,
                #           complevel=_STORE_COMP_LVL,
                #           )

                stores[entry] = df

                if verbose:
                    iterable.set_postfix({
                        'step': f'{entry.step}',
                        'data_key': f'{entry.data_key}',
                        'location_id': f'({entry.src_key}, {entry.loc_id})',
                        # 'coordinates': f'({entry.lat}, {entry.lon})',
                        'year': f'{entry.year}',
                        # 'location': f'{entry.loc_name} ({entry.country_code})',
                        # 'success': f'{success}',
                        # 'attempt': f'{num_attempts}',
                        'n_failed': f'{num_failed}/{len(entries)}',
                    })

            except OpenMeteoRequestsError as exception:
                num_failed += 1

                print(exception)
                print(repr(exception))

                if verbose:
                    iterable.set_postfix({
                        'step': f'{entry.step}',
                        'data_key': f'{entry.data_key}',
                        'location_id': f'({entry.src_key}, {entry.loc_id})',
                        # 'coordinates': f'({entry.lat}, {entry.lon})',
                        'year': f'{entry.year}',
                        # 'location': f'{entry.loc_name} ({entry.country_code})',
                        'error': f'{repr(exception)}',
                        'n_failed': f'{num_failed}/{len(entries)}',
                    })

            time.sleep(_DOWNLOAD_DELAY)

    return dict()  # TODO -- info dict


def _get_base_url(commercial: bool = False) -> str:
    return f"https://{'customer-' if commercial else ''}archive-api.open-meteo.com/v1/archive"


def _get_api_key():
    if os.path.exists(_PATH_API_KEY):
        with open(_PATH_API_KEY) as f:
            key = f.read()
            return key
    else:
        return None


# # List data types for which we use different keys than the openmeteo API. Map them to the key used by openmeteo
# _DATA_KEY_EXCEPTIONS = {  # Translate data keys to those used by openmeteo
#     config.KEY_FEATURE_TEMPERATURE: 'temperature_2m',
#     config.KEY_FEATURE_ISDAY: 'is_day',
#     config.KEY_FEATURE_PRECIPITATION: 'precipitation',
# }


class OpenMeteoDownloadException(Exception):
    pass  # For throwing errors if anything goes wrong during the download process


if __name__ == '__main__':

    _dly_keys = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "daylight_duration",
    ]
    _hly_keys = [
        'temperature_2m',
        'is_day',
        'precipitation',
    ]
    _skps_hly = list(zip(['hourly'] * len(_hly_keys), _hly_keys))
    _skps_dly = list(zip(['daily'] * len(_dly_keys), _dly_keys))
    _skps = _skps_hly + _skps_dly

    _stores = OpenMeteoStores(
        step_key_pairs=_skps
    )
    _stores._transfer_store()


