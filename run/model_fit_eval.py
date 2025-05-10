import argparse
import os

from run.args_util.dataset_init import configure_argparser_dataset, configure_argparser_dataset_split, \
    load_dataset_from_args, split_dataset_from_args
from run.args_util.eval import evaluate_model_using_args
from run.args_util.model_init import MODEL_STR_OPTIONS, configure_argparser_model, obtain_model_from_args

"""
    Helper functions
"""


def _configure_argparser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # Configure arguments related to main flow of the run

    parser.add_argument('--run_name',
                        type=str,
                        required=True,
                        )

    parser.add_argument('--model_name',
                        type=str,
                        required=False,
                        )

    parser.add_argument('--model_cls',
                        type=str,
                        choices=MODEL_STR_OPTIONS,
                        required=True,
                        )

    # parser.add_argument('--target',
    #                     type=str,
    #                     required=True,
    #                     )

    parser.add_argument('--seed',
                        type=int,
                        required=True,
                        )

    # Parse known args to configure parser based on specified args (e.g. model hyperparameters depending on model cls)
    known_args, _ = parser.parse_known_args()
    model_cls = known_args.model_cls

    configure_argparser_dataset(parser)
    configure_argparser_dataset_split(parser)
    configure_argparser_model(parser, model_cls)

    return parser


DATASET_KEY_TO_TARGET = {
    'GMU_Cherry_Japan': 'gmu_0',
    'GMU_Cherry_Switzerland': 'gmu_1',
    'GMU_Cherry_South_Korea': 'gmu_2',
    'PEP725_Apple': 'BBCH_60',
    'PEP725_Pear': 'BBCH_60',
    'PEP725_Peach': 'BBCH_60',
    'PEP725_Almond': 'BBCH_60',
    'PEP725_Hazel': 'BBCH_60',
    'PEP725_Cherry': 'BBCH_60',
    'PEP725_Apricot': 'BBCH_60',
    'PEP725_Blackthorn': 'BBCH_60',
    'PEP725_Plum': 'BBCH_60',
}


"""
    Script

    1. Parse arguments configuring the experiment
    2. Load the configured dataset
    3. Split the dataset in train and test
    4. Create and fit a model, as specified in the arguments
    5. Evaluate the model
    6. Save results and plots, finalize the run

"""

if __name__ == '__main__':

    """
        1. Create/configure argument parser
    """
    parser = argparse.ArgumentParser()
    _configure_argparser(parser)

    args = parser.parse_args()
    args.target = DATASET_KEY_TO_TARGET[args.dataset_name]

    """
        2. Create/load the dataset
    """
    dataset = load_dataset_from_args(args)

    """
        3. Split dataset in train/val/test
    """
    ds_trn, ds_tst = split_dataset_from_args(dataset,
                                             args,
                                             )

    """
        4. Create and fit a model
    """

    model = obtain_model_from_args(args, ds_trn)

    """
        5. Evaluate the model
    """
    eval_result = evaluate_model_using_args(model,
                                            ds_trn,
                                            ds_tst,
                                            args,
                                            )

    """
        6. Save results and plots, finalize the run
    """

    path_base = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(
        path_base,
        'figures',
        'model_fit_eval',
        args.dataset_name,
        args.model_cls,
        # args.model_name,
    )

    eval_result.save()
    eval_result.savefig_scatter(path=path, fn='scatter.png', as_doy=True)

    dfs_metrics = eval_result.compute_metrics()

    print('\nTrain')
    print(dfs_metrics['train'])
    print('\nTest')
    print(dfs_metrics['test'])

    model.save(args.model_name)

    dataset.close()
