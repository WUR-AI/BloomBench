import os

import geopandas as gpd
from geopandas import GeoDataFrame

from phenology.config import PATH_DATA

_PATH_ADMIN_BOUNDARIES = os.path.join(PATH_DATA, 'resources', 'world-administrative-boundaries.geojson')


def load_admin_boundaries() -> GeoDataFrame:
    gdf = gpd.read_file(_PATH_ADMIN_BOUNDARIES,
                        # driver='GeoJSON',
                        )
    # print(gdf.crs)
    return gdf


if __name__ == '__main__':

    # Data source: https://public.opendatasoft.com/explore/dataset/world-administrative-boundaries/export/

    gdf = load_admin_boundaries()

    print(type(gdf['geo_point_2d']))

    print(gdf.columns)
