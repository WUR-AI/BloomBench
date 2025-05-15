
import os

import numpy as np


"""
    Project Structure Configuration
"""

# Project root directory
PATH_BASE = os.path.abspath(os.path.dirname(__file__))

# Path to directory where data is stored
PATH_DATA = os.path.join(PATH_BASE, 'data')
os.makedirs(PATH_DATA, exist_ok=True)

# Path to directory where baseline models are stored
PATH_MODELS = os.path.join(PATH_DATA, 'models', 'saved_models')
os.makedirs(PATH_MODELS, exist_ok=True)

# Path to directory where evaluation output is stored
PATH_OUTPUT_EVAL = os.path.join(PATH_BASE, 'evaluation', 'model_evaluation')

"""
    Project Configuration
"""
# Max nr of samples that will be cached
SAMPLE_CACHE_SIZE = 100000

# RNG seeds
SEED_DATA_PREPROCESSING = 123
RNG_DATA_PREPROCESSING = np.random.default_rng(seed=SEED_DATA_PREPROCESSING)

"""
    Global constants
"""
# Data source key (i.e. short string to characterize the data source)
KEY_PEP725 = 'pep725'
KEY_GMU_CHERRY = 'GMU_cherry'

# All target data source keys
KEYS_DATA_SOURCES = [
    KEY_PEP725,
    KEY_GMU_CHERRY,
]

KEY_OBSERVATIONS = 'observations'
KEY_OBSERVATIONS_INDEX = 'observations_index'
KEY_FEATURES = 'features'

KEY_DATA_SOURCE = 'src'
KEY_LOC_ID = 'loc_id'
KEY_YEAR = 'year'
KEY_SPECIES_CODE = 'species_code'
KEY_SUBGROUP_CODE = 'subgroup_code'
KEY_OBS_TYPE = 'obs_type'

KEY_LAT = 'lat'
KEY_LON = 'lon'
KEY_ALT = 'alt'
KEY_LOC_NAME = 'loc_name'
KEY_COUNTRY_CODE = 'country_code'

KEYS_INDEX = (
    KEY_DATA_SOURCE,
    KEY_LOC_ID,
    KEY_YEAR,
    KEY_SPECIES_CODE,
    KEY_SUBGROUP_CODE,
)

COUNTRY_CODE_NAME = {  # TODO -- align with PEP? -- iso3?
    'AT': 'Austria',
    'BA': 'Bosnia and Herzegovina',
    'BE': 'Belgium',
    'CZ': 'Czech Republic',
    'DE': 'Germany',
    'ES': 'Spain',
    'HR': 'Croatia',
    'ME': 'Montenegrin Republic',
    'LT': 'Lithuania',
    'SK': 'Slovakia',
    'FR': 'France',
    'JP': 'Japan',
    'SW': 'Switzerland',
    'KR': 'South Korea',
}
COUNTRY_CODES = tuple(COUNTRY_CODE_NAME.keys())

# Lower bound on grid spatial resolution
MIN_GRID_SIZE = (0.25, 0.25)  # Degrees (lat, lon)  ERA5 GRID

"""
    Data Configuration
"""

# Size of the grid used to "prune" the observation data. That is, from each grid cell one observation will be selected
AGG_GRID_SIZE = MIN_GRID_SIZE

# Validate AGG_GRID_SIZE
assert len(AGG_GRID_SIZE) == 2
# Make sure the grid is at least as big as the ERA5 grid
# Otherwise data leakage can occur if two observations originate from the same cell
assert all([v >= w for v, w in zip(AGG_GRID_SIZE, MIN_GRID_SIZE)])


