import os

import phenology.config

PATH_OPENMETEO = os.path.join(phenology.config.PATH_DATA, 'openmeteo')
os.makedirs(PATH_OPENMETEO, exist_ok=True)

PATH_OPENMETEO_DATA = os.path.join(PATH_OPENMETEO, 'data')
os.makedirs(PATH_OPENMETEO_DATA, exist_ok=True)