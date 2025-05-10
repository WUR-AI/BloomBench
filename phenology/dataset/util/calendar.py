from collections import defaultdict
from functools import lru_cache

import numpy as np

from phenology.config import KEY_PEP725, KEY_GMU_CHERRY


class Calendar:

    def __init__(self, calendar: dict = None):
        # Season (start date (MM-DD), length) for all species
        # TODO -- this is not only crop dependent but also location!
        # TODO -- read from csv!
        self._calendar = calendar or defaultdict(lambda: ('10-01', 365))

        # src, loc_id, year, species_code, subgroup_code

    @lru_cache(maxsize=None)  # Cache least-recently-used/requested data
    def get_season_info(self,
                        year: int,
                        src: str,
                        species_code,
                        subgroup_code=None,  # TODO
                        loc_id: str = None,  # TODO
                        ) -> dict:

        start_date, length = self._calendar[src, species_code, subgroup_code]

        # Perform check if end of season is in next year
        start_date = np.datetime64(f'{year - 1}-{start_date}')
        td = np.timedelta64(length, 'D')
        end_date = start_date + td

        # TODO -- there's an edge case where this breaks due to leap years. For now its no problem however™
        if end_date.astype('datetime64[Y]').astype(int) + 1970 != year:
            start_date = np.datetime64(f'{year}-{start_date}')
            td = np.timedelta64(length, 'D')
            end_date = start_date + td

        return {
            'season_start': start_date,
            'season_end': end_date,
            'season_length': td,
            'year': year,
        }

    def set_season_info(self,
                        end_date: str,
                        length: int,
                        src: str,
                        species_code,
                        subgroup_code=None,
                        loc_id: str=None,
                        year: int=None,
                        ):

        self._calendar[src, species_code, subgroup_code] = end_date, length

    @classmethod
    def from_config(cls) -> 'Calendar':
        return Calendar()  # TODO

