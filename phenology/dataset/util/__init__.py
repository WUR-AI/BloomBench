

import numpy as np
import pandas as pd


def doy_to_date_in_year(year: int, doy: int) -> np.datetime64:
    assert 0 < doy <= 365
    return np.datetime64(f'{year}-01-01') + np.timedelta64(doy - 1, 'D')


# def date_to_doy(date: np.datetime64) -> int:
#
#     # (np.datetime64(f'{year}-12-31') - start) // np.timedelta64(1, 'D'),
#
#     raise NotImplementedError   # TODO
#
#
# def doy_to_index(doy: int,
#                  src: str,
#                  species_code: int,
#                  subgroup_code: int,
#                  year: int,
#                  prev_year: bool = False,
#                  ) -> int:
#
#     # Obtain season start date, season length for the plant species
#     ss, sl = config.SPECIES_SEASON[src, species_code, subgroup_code]
#
#     # Convert start date to timestamp in the respective year
#     ss = pd.Timestamp(f'{year - 1}-{ss}')
#
#     if prev_year:
#         ix = doy - ss.dayofyear
#         return ix
#     else:
#         ts = pd.Timestamp(f'{year}-01-01') + pd.Timedelta(days=doy - 1)
#         ix = (ts - ss).days
#         return ix
#
#
# def index_to_doy(index: int,
#                  src: str,
#                  species_code: int,
#                  subgroup_code: int,
#                  year: int,
#                  ) -> int:
#     """
#     Convert in-season index to a DOY in the respective year
#     :param index:
#     :param src:
#     :param species_code:
#     :param subgroup_code:
#     :param year:
#     :return:
#     """
#     assert index >= 0
#     # Obtain season start date, season length for the plant species
#     ss, sl = config.SPECIES_SEASON[src, species_code, subgroup_code]
#     # Convert to timestamp in the previous year
#     # TODO -- this assumes season start is in previous year!
#     # TODO -- this is true for the current research, but should  be more general!
#     # TODO -- year offset can be computed by getting the season end date and comparing years
#     ss = pd.Timestamp(f'{year - 1}-{ss}')
#     # Obtain DOY from timestamp after adding the required nr of days
#     date = ss + pd.Timedelta(days=index)
#     return date.dayofyear


class DatasetException(Exception):
    pass

