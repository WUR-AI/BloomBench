import argparse

from phenology.dataset.dataset import Dataset


def _configure_argparser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--dataset_time_step',
                        type=str,
                        required=True,
                        choices=['daily', 'hourly'],
                        )
    parser.add_argument('--data_keys',
                        nargs='*',
                        )
    parser.add_argument('--dataset_names',
                        nargs='+',
                        )

    return parser


if __name__ == '__main__':

    """
        Script for obtaining all required data

        Obtains data by iterating over the datasets and checking feature data availability for the observations

        See 
        https://open-meteo.com/en/docs/historical-weather-api
        for key names
    """

    parser = argparse.ArgumentParser(description='Script for obtaining all required data')
    parser = _configure_argparser(parser)
    args = parser.parse_args()

    if args.dataset_names is None:
        dataset_names = [
            'GMU_Cherry_Japan',
            'GMU_Cherry_Switzerland',
            'GMU_Cherry_South_Korea',
            'PEP725_Apple',
            'PEP725_Pear',
            'PEP725_Peach',
            'PEP725_Almond',
            'PEP725_Hazel',
            'PEP725_Cherry',
            'PEP725_Apricot',
            'PEP725_Blackthorn',
            'PEP725_Plum',
        ]
    else:
        dataset_names = list(args.dataset_names)

    step = args.dataset_time_step

    # If data keys were not specified -> set to default depending on time step
    if args.data_keys is None:

        step_to_data_keys = {
            'hourly': [
                'temperature_2m',
                'is_day',
                # 'precipitation',
                # 'relative_humidity_2m',
                # 'soil_moisture_0_to_7cm',
                # 'soil_moisture_7_to_28cm',
                # 'shortwave_radiation',
            ],
            'daily': [
                # "temperature_2m_max",
                # "temperature_2m_min",
                "temperature_2m_mean",
                "daylight_duration",
                # "sunshine_duration",
                # "precipitation_sum",
                # "shortwave_radiation_sum",
            ],
        }

        data_keys = step_to_data_keys[step]
    else:
        data_keys = list(args.data_keys)

    # Load all datasets once
    for dataset_name in dataset_names:
        print(f'Obtaining data for "{dataset_name}"')
        with Dataset.load(dataset_name,
                          step=step,
                          data_keys=data_keys,
                          ) as dataset:
            pass
