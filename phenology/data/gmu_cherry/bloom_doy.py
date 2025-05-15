
import pandas as pd

from unidecode import unidecode

from phenology.data.gmu_cherry.download import *


"""

    Code for pre-processing the cherry blossom DOY data 

"""


# COUNTRIES = ('Japan', 'South Korea', 'Switzerland', 'USA')
COUNTRIES = ('Japan', 'South Korea', 'Switzerland',)


def get_df_japan() -> pd.DataFrame:

    # Get the japan dataset
    df_japan = get_data_japan()
    # Sanitize the location labels
    df_japan['location'] = df_japan['location'].apply(_sanitize_location_label)
    # Japan's Akita data is duplicated. Remove the duplicates
    df_japan = df_japan.drop_duplicates()

    # The Kushiro location contains data for two separate locations (at 4.5 alt and 14.05 alt)
    # Separate the Kushiro data into two locations
    df_japan.loc[(df_japan['location'] == 'Japan/Kushiro') & (df_japan['alt'] == 4.5), 'location'] = 'Japan/Kushiro_1'
    df_japan.loc[(df_japan['location'] == 'Japan/Kushiro') & (df_japan['alt'] == 14.05), 'location'] = 'Japan/Kushiro_2'
    # Do the same for Muroran
    df_japan.loc[(df_japan['location'] == 'Japan/Muroran') & (df_japan['alt'] == 39.89), 'location'] = 'Japan/Muroran_1'
    df_japan.loc[(df_japan['location'] == 'Japan/Muroran') & (df_japan['alt'] == 3.00), 'location'] = 'Japan/Muroran_2'
    # And Sendai
    df_japan.loc[(df_japan['location'] == 'Japan/Sendai') & (df_japan['alt'] == 38.85), 'location'] = 'Japan/Sendai_1'
    df_japan.loc[(df_japan['location'] == 'Japan/Sendai') & (df_japan['alt'] == 37.90), 'location'] = 'Japan/Sendai_2'
    # And Nagoya
    df_japan.loc[(df_japan['location'] == 'Japan/Nagoya') & (df_japan['lat'] < 35.168), 'location'] = 'Japan/Nagoya_1'
    df_japan.loc[(df_japan['location'] == 'Japan/Nagoya') & (df_japan['lat'] > 35.168), 'location'] = 'Japan/Nagoya_2'
    # And Tottori
    df_japan.loc[(df_japan['location'] == 'Japan/Tottori') & (df_japan['alt'] == 7.1), 'location'] = 'Japan/Tottori_1'
    df_japan.loc[(df_japan['location'] == 'Japan/Tottori') & (df_japan['alt'] == 6.0), 'location'] = 'Japan/Tottori_2'
    # And Izuhara
    df_japan.loc[(df_japan['location'] == 'Japan/Izuhara') & (df_japan['alt'] == 3.65), 'location'] = 'Japan/Izuhara_1'
    df_japan.loc[(df_japan['location'] == 'Japan/Izuhara') & (df_japan['alt'] == 130.00), 'location'] = 'Japan/Izuhara_2'
    # And Yakushima
    df_japan.loc[(df_japan['location'] == 'Japan/Yakushima') & (df_japan['alt'] == 37.3), 'location'] = 'Japan/Yakushima_1'
    df_japan.loc[(df_japan['location'] == 'Japan/Yakushima') & (df_japan['alt'] == 36.0), 'location'] = 'Japan/Yakushima_2'
    # And Kochi
    df_japan.loc[(df_japan['location'] == 'Japan/Kochi') & (df_japan['alt'] == 0.5), 'location'] = 'Japan/Kochi_1'
    df_japan.loc[(df_japan['location'] == 'Japan/Kochi') & (df_japan['alt'] == 3.0), 'location'] = 'Japan/Kochi_2'

    # Get the Kyoto dataset (overlaps with japan)
    df_kyoto = get_data_kyoto()
    # Overwrite location labels to match the format
    df_kyoto['location'] = df_kyoto['location'].apply(lambda x: 'Japan/Kyoto_2')
    # Rename the existing Kyoto entries of the japan dataset
    df_japan['location'] = df_japan['location'].replace('Japan/Kyoto', 'Japan/Kyoto_1')

    # Add Kyoto to the Japan dataframe
    df_japan = pd.concat([df_japan, df_kyoto])

    return df_japan


def get_df_switzerland() -> pd.DataFrame:
    # Get the swiss dataset
    df_swiss = get_data_meteoswiss()
    # Sanitize the location labels
    df_swiss['location'] = df_swiss['location'].apply(_sanitize_location_label)

    # Get the liestal data
    df_liestal = get_data_liestal()
    # Overwrite location labels to match the format
    df_liestal['location'] = df_liestal['location'].apply(lambda x: 'Switzerland/Liestal_2')
    # Rename the existing Liestal entries of the swiss dataset
    df_swiss['location'] = df_swiss['location'].replace('Switzerland/Liestal', 'Switzerland/Liestal_1')
    # Add liestal to the swiss dataframe
    df_swiss = pd.concat([df_swiss, df_liestal])

    return df_swiss


def get_df_south_korea() -> pd.DataFrame:
    # Get the south korea data
    df_south_korea = get_data_south_korea()
    # Sanitize the location labels
    df_south_korea['location'] = df_south_korea['location'].apply(_sanitize_location_label)

    return df_south_korea


# def get_df_usa() -> pd.DataFrame:
#     # Get the washington data
#     df_usa = get_data_washingtondc()
#     # Overwrite location labels to match the format
#     df_usa['location'] = df_usa['location'].apply(lambda x: 'USA/Washington_DC')
#
#     return df_usa


# def get_df(set_index: bool = False):
#
#     df_japan = get_df_japan()
#     df_swiss = get_df_switzerland()
#     df_south_korea = get_df_south_korea()
#     df_usa = get_df_usa()
#
#     # Combine all dataframes
#     df = pd.concat([
#         df_japan,
#         df_swiss,
#         df_south_korea,
#         df_usa,
#     ])
#
#     if set_index:
#         # Entries are uniquely indexed by their year and location
#         df.set_index(['year', 'location'], inplace=True)
#     else:
#         df.reset_index(inplace=True)
#
#     return df


def get_locations_japan() -> list:
    df = get_df_japan()
    locs = df['location'].values
    return list(set(locs))


def get_locations_coordinates_japan() -> pd.DataFrame:
    df = get_data_japan()
    df.set_index('location', inplace=True)
    df = df[['lat', 'long', 'alt']]
    df.drop_duplicates(inplace=True)
    return df


def get_locations_switzerland() -> list:
    df = get_df_switzerland()
    locs = df['location'].values
    return list(set(locs))


def get_locations_coordinates_switzerland() -> pd.DataFrame:
    df = get_df_switzerland()
    df.set_index('location', inplace=True)
    df = df[['lat', 'long', 'alt']]
    df.drop_duplicates(inplace=True)
    return df


def get_locations_south_korea() -> list:
    df = get_df_south_korea()
    locs = df['location'].values
    return list(set(locs))


def get_locations_coordinates_south_korea() -> pd.DataFrame:
    df = get_df_south_korea()
    df.set_index('location', inplace=True)
    df = df[['lat', 'long', 'alt']]
    df.drop_duplicates(inplace=True)
    return df


def _sanitize_location_label(s: str) -> str:  # TODO -- redefine
    tokens = s.split('/')

    if len(tokens) < 2:
        raise Exception(f'Could not parse location entry "{s}"')
    # The first token is the country name
    country = tokens[0]
    country = country.replace(' ', '')
    country = country.replace('-', '')
    # The remaining tokens form the location name
    location = ','.join(tokens[1:])
    location = location.replace(' ', '')
    location = location.replace('-', '_')
    location = location.replace('\'', '_')
    location = location.replace(',', '_')
    location = location.replace('.', '_')
    location = location.replace('(', '_')
    location = location.replace(')', '_')

    location = unidecode(location)

    return f'{country}/{location}'


# def get_locations_usa() -> list:
#     df = get_df_usa()
#     locs = df['location'].values
#     return list(set(locs))
#
#
# def get_locations_coordinates_usa() -> pd.DataFrame:
#     df = get_df_usa()
#     df.set_index('location', inplace=True)
#     df = df[['lat', 'long', 'alt']]
#     df.drop_duplicates(inplace=True)
#     return df


# def get_locations() -> list:
#     return get_locations_japan() + get_locations_switzerland() + get_locations_south_korea() + get_locations_usa()


# def get_locations_coordinates() -> pd.DataFrame:
#
#     df = pd.concat([
#         get_locations_coordinates_japan(),
#         get_locations_coordinates_switzerland(),
#         get_locations_coordinates_south_korea(),
#         get_locations_coordinates_usa(),
#     ])
#
#     return df

#
# if __name__ == '__main__':
#
#     # print(get_locations_coordinates_japan().to_dict(orient='index'))
#
#     _data = get_df(set_index=True)
#     #
#     # print(_data.columns)
#     # print()
#     print(_data)
#     print(_data.iloc[_data.index.get_level_values('year') == 2000])




