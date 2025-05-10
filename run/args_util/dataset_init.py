import argparse
from typing import Tuple

import sklearn

from phenology.dataset.dataset import Dataset
from run.args_util import ExperimentConfigException


def configure_argparser_dataset(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    parser.add_argument('--dataset_name',
                        type=str,
                        required=True,
                        )

    parser.add_argument('--dataset_time_step',
                        type=str,
                        default='daily',
                        )

    parser.add_argument('--data_keys',
                        nargs='*',
                        required=True,
                        )

    # parser.add_argument('--data_cache_save',
    #                     action='store_true',
    #                     )
    #
    # parser.add_argument('--data_cache_load',
    #                     action='store_true',
    #                     )

    parser.add_argument('--skip_meteo_data_check',
                        action='store_true',
                        )

    return parser


def load_dataset_from_args(args: argparse.Namespace) -> Dataset:

    dataset_name = args.dataset_name
    dataset_time_step = args.dataset_time_step
    data_keys = args.data_keys

    dataset = Dataset.load(dataset_name,
                           step=dataset_time_step,
                           data_keys=data_keys,
                           mode_download_meteo='skip' if args.skip_meteo_data_check else None,
                           # load_cache_meteo=args.data_cache_load,
                           # save_meteo_data=args.data_cache_save,
                           )

    return dataset


def configure_argparser_dataset_split(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    # Dataset split args
    # parser.add_argument('--split_years',
    #                     action='store_true',
    #                     )

    # parser.add_argument('--split_locations',
    #                     action='store_true',
    #                     )

    parser.add_argument('--split_years_size',
                        type=float,
                        default=0.75,
                        )

    # parser.add_argument('--split_locations_size',
    #                     type=float,
    #                     default=0.75,
    #                     )
    # parser.add_argument('--split_locations_grid_size',
    #                     type=float,
    #                     default=1,
    #                     )

    parser.add_argument('--seed_data_split',
                        type=int,
                        required=False,
                        )

    return parser


def split_dataset_from_args(dataset: Dataset,
                            args: argparse.Namespace,
                            ) -> Tuple[Dataset, Dataset]:

    if args.seed_data_split is not None:
        seed = args.seed_data_split
    else:
        seed = args.seed

    years = dataset.years
    year_min = min(years)
    year_max = max(years)
    years_complete = list(range(year_min, year_max + 1))

    # method = 'random'
    method = 'cutoff'

    match method:

        case 'random':
            # Perform data split
            years_trn, years_tst = sklearn.model_selection.train_test_split(years_complete,
                                                                            train_size=args.split_years_size,
                                                                            shuffle=True,
                                                                            random_state=seed,
                                                                            )

            dataset_trn = dataset.select_years(years_trn)
            dataset_tst = dataset.select_years(years_tst)

        case 'cutoff':
            # Choose cutoff point
            ix = int(len(years_complete) * args.split_years_size)
            years_trn = years_complete[:ix]
            years_tst = years_complete[ix:]
            dataset_trn = dataset.select_years(years_trn)
            dataset_tst = dataset.select_years(years_tst)

        case _:
            raise ExperimentConfigException('Undefined data split procedure: {}'.format(method))

    return dataset_trn, dataset_tst


# def split_dataset_from_args(dataset: Dataset,
#                             args: argparse.Namespace,
#                             ) -> Tuple[Dataset, Dataset, Dataset]:
#
#     if args.seed_data_split is not None:
#         seed = args.seed_data_split
#     else:
#         seed = args.seed
#
#     if args.split_locations:
#
#         split_locations_size = args.split_locations_size
#         grid_size = args.split_locations_grid_size
#
#         # Split dataset in train/test
#         dataset_trn, dataset_tst, _ = dataset.split_by_grid(
#             grid_size=(grid_size, grid_size),
#             split_size=split_locations_size,
#             shuffle=True,
#             random_state=seed,
#         )
#         # Split train dataset in train/val
#         dataset_trn, dataset_val, _ = dataset_trn.split_by_grid(
#             grid_size=(grid_size, grid_size),
#             split_size=split_locations_size,
#             shuffle=True,
#             random_state=seed,
#         )
#         return dataset_trn, dataset_val, dataset_tst
#     else:
#         return dataset, dataset, dataset
