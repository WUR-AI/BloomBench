import argparse

import torch

from phenology.dataset.dataset import Dataset
from phenology.models.adaboost import AdaBoostModel
from phenology.models.base import NullModel, BaseModel
from phenology.models.base_torch import NullTorchModel
from phenology.models.gradient_boosting import GradientBoostingModel
from phenology.models.gru import GRUModel
from phenology.models.linear import LinearTrendModel
from phenology.models.lstm import LSTMModel
from phenology.models.mean import MeanModel
from phenology.models.random_forest import RandomForestModel
from run.args_util import ExperimentConfigException

MODEL_CLS_OPTIONS = (
    NullModel,  # For testing
    MeanModel,
    LinearTrendModel,
    RandomForestModel,
    GradientBoostingModel,
    AdaBoostModel,
    NullTorchModel,  # For testing
    LSTMModel,
    GRUModel,
)

MODEL_STR_OPTIONS = tuple(
    cls.__name__ for cls in MODEL_CLS_OPTIONS
)


_MODEL_STR_TO_CLS = {
    cls.__name__: cls for cls in MODEL_CLS_OPTIONS
}


def configure_argparser_model(parser: argparse.ArgumentParser, model_cls_name: str) -> argparse.ArgumentParser:
    """
    Function that configures the argument parser depending on which model class has been selected
    :param parser: the argument parser to configure
    :param model_cls_name: the name of the model class
    :return: the configured argument parser
    """

    parser.add_argument('--load_model',
                        action='store_true',
                        )

    model_cls = _MODEL_STR_TO_CLS[model_cls_name]

    match model_cls_name:  # TODO -- implement as classmethod of BaseModel that is supposed to be overwritten?

        case NullModel.__name__:
            pass

        case MeanModel.__name__:
            pass

        case LinearTrendModel.__name__:
            pass

        case RandomForestModel.__name__:
            _configure_argparser_random_forest(parser)

        case GradientBoostingModel.__name__:
            _configure_argparser_gradient_boosting(parser)

        case AdaBoostModel.__name__:
            _configure_argparser_adaboost(parser)

        case NullTorchModel.__name__:
            _configure_argparser_fit_torch(parser)

        case LSTMModel.__name__:
            _configure_argparser_fit_torch(parser)

        case GRUModel.__name__:
            _configure_argparser_fit_torch(parser)

        case _:
            raise ExperimentConfigException(f'Model class "{model_cls_name}" not recognized while parsing arguments')

    return parser


def _configure_argparser_random_forest(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    parser.add_argument('--fit_hyperparameters',
                        action='store_true',
                        )

    parser.add_argument('--hf_n_iter',
                        type=int,
                        default=10,
                        help='Number of hyperparameter optimization iterations'
                        )

    parser.add_argument('--hf_cv',
                        type=int,
                        default=5,
                        help='Number of folds used when optimizing hyperparameters')

    return parser


def _configure_argparser_gradient_boosting(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    parser.add_argument('--fit_hyperparameters',
                        action='store_true',
                        )

    parser.add_argument('--hf_n_iter',
                        type=int,
                        default=10,
                        help='Number of hyperparameter optimization iterations'
                        )

    parser.add_argument('--hf_cv',
                        type=int,
                        default=5,
                        help='Number of folds used when optimizing hyperparameters')

    return parser


def _configure_argparser_adaboost(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    parser.add_argument('--fit_hyperparameters',
                        action='store_true',
                        )

    parser.add_argument('--hf_n_iter',
                        type=int,
                        default=10,
                        help='Number of hyperparameter optimization iterations'
                        )

    parser.add_argument('--hf_cv',
                        type=int,
                        default=5,
                        help='Number of folds used when optimizing hyperparameters')

    return parser


TORCH_OPTIMIZERS = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
}


def _configure_argparser_fit_torch(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Configure the provided argument parser to parse arguments that relate to training a PyTorch model
    The configuration is in-place
    :param parser: the parser that should be configured
    :return: a reference to the provided parser, now configured
    """
    parser.add_argument('--num_epochs',
                        type=int,
                        default=1,
                        help='Number of iterations of training over the entire training dataset',
                        )
    parser.add_argument('--batch_size',
                        type=int,
                        default=None,
                        help='Batch size used during the optimization process. Default is None, meaning that '
                             'gradients will be computed over the entire dataset',
                        )
    parser.add_argument('--scheduler_step_size',
                        type=int,
                        default=None,
                        help='Learning rate is decayed every `scheduler_step_size` iterations. If not specified, '
                             'the learning rate will never be decayed',
                        )
    parser.add_argument('--scheduler_decay',
                        type=float,
                        default=0.5,
                        help='Factor by which the learning rate will be scaled when decaying the learning rate',
                        )
    parser.add_argument('--clip_gradient',
                        type=float,
                        default=None,
                        help='Value by which the gradient values will be clipped. No clipping is done when left '
                             'unspecified',
                        )
    parser.add_argument('--validation_period',
                        type=int,
                        default=None,
                        help='Number of epochs between validation steps. No validation is done when left unspecified',
                        )
    parser.add_argument('--optimizer',
                        type=str,
                        choices=list(TORCH_OPTIMIZERS.keys()),
                        default='sgd',
                        help='Optimizer that is used when fitting the model',
                        )
    parser.add_argument('--lr',
                        type=float,
                        default=1e-2,
                        help='Learning rate',
                        )
    parser.add_argument('--weight_decay',
                        type=float,
                        default=None,
                        help='Optimizer weight decay regularization',
                        )
    parser.add_argument('--early_stopping',
                        action='store_true',
                        help='Flag indicating whether an early stopping criterion should be used when fitting. The '
                             'criterion used is: validation_loss > (self.min_validation_loss + self.min_delta), where '
                             'min_validation_loss is the minimum validation loss that was encountered so far and '
                             'min_delta is the required difference between the current validation loss and the '
                             'minimum validation loss.',
                        )
    parser.add_argument('--early_stopping_patience',
                        type=int,
                        default=1,
                        help='Patience for early stopping (i.e. nr of times the criterion needs to hold in succession '
                             'before stopping)',
                        )
    parser.add_argument('--early_stopping_min_delta',
                        type=float,
                        default=0,
                        help='Minimum required difference between validation loss and minimum validation loss.',
                        )

    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that controls whether GPU usage should be disabled',
                        )

    return parser


def obtain_model_from_args(args: argparse.Namespace, dataset: Dataset) -> BaseModel:

    model_cls_name = args.model_cls
    model_cls = _MODEL_STR_TO_CLS[model_cls_name]
    model_name = args.model_name if args.model_name is not None else model_cls.__name__

    target = args.target

    if args.load_model:
        model, _ = model_cls.load(model_name)
        return model

    # dataset.cache_data(verbose=True, desc='Loading training data in memory')

    match model_cls_name:

        case RandomForestModel.__name__:

            model_kwargs = None if args.fit_hyperparameters else _map_kwargs_random_forest(args.dataset_name)

            model, _ = RandomForestModel.fit(
                target_key=target,
                dataset=dataset,
                model_name=model_name,
                hf=args.fit_hyperparameters,
                hf_n_iter=args.hf_n_iter,
                hf_cv=args.hf_cv,
                hf_random_state=args.seed,
                model_kwargs={
                    'random_forest_kwargs': model_kwargs,
                },
            )

        case GradientBoostingModel.__name__:

            model_kwargs = None if args.fit_hyperparameters else _map_kwargs_gradient_boosting(args.dataset_name)

            model, _ = GradientBoostingModel.fit(
                target_key=target,
                dataset=dataset,
                model_name=model_name,
                hf=args.fit_hyperparameters,
                hf_n_iter=args.hf_n_iter,
                hf_cv=args.hf_cv,
                hf_random_state=args.seed,
                model_kwargs={
                    'gradient_boosting_kwargs': model_kwargs,
                },
            )

        case AdaBoostModel.__name__:

            model_kwargs = None if args.fit_hyperparameters else _map_kwargs_adaboost(args.dataset_name)

            model, _ = AdaBoostModel.fit(
                target_key=target,
                dataset=dataset,
                model_name=model_name,
                hf=args.fit_hyperparameters,
                hf_n_iter=args.hf_n_iter,
                hf_cv=args.hf_cv,
                hf_random_state=args.seed,
                model_kwargs={
                    'adaboost_kwargs': model_kwargs,
                },
            )

        case NullTorchModel.__name__:

            device = torch.device('cuda' if torch.cuda.is_available() and not args.disable_cuda else 'cpu')

            model, _ = NullTorchModel.fit(
                target_key=target,
                dataset=dataset,
                model_name=model_name,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                scheduler_step_size=args.scheduler_step_size,
                scheduler_decay=args.scheduler_decay,
                clip_gradient=args.clip_gradient,
                val_period=args.validation_period,
                optimizer=args.optimizer,
                optimizer_kwargs={
                    'lr': args.lr,
                    **({'weight_decay': args.weight_decay} if args.weight_decay is not None else {}),
                },
                early_stopping=args.early_stopping,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                model_kwargs={
                    'step_size': args.dataset_time_step,
                    'data_keys': args.data_keys,
                },
                seed=args.seed,
                device=device,
            )

        case LSTMModel.__name__:

            device = torch.device('cuda' if torch.cuda.is_available() and not args.disable_cuda else 'cpu')

            model, _ = LSTMModel.fit(
                target_key=target,
                dataset=dataset,
                model_name=model_name,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                scheduler_step_size=args.scheduler_step_size,
                scheduler_decay=args.scheduler_decay,
                clip_gradient=args.clip_gradient,
                val_period=args.validation_period,
                optimizer=args.optimizer,
                optimizer_kwargs={
                    'lr': args.lr,
                    **({'weight_decay': args.weight_decay} if args.weight_decay is not None else {}),
                },
                early_stopping=args.early_stopping,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                model_kwargs={
                    'step_size': args.dataset_time_step,
                    'data_keys': args.data_keys,
                },
                seed=args.seed,
                device=device,
            )

        case GRUModel.__name__:

            device = torch.device('cuda' if torch.cuda.is_available() and not args.disable_cuda else 'cpu')

            model, _ = LSTMModel.fit(
                target_key=target,
                dataset=dataset,
                model_name=model_name,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                scheduler_step_size=args.scheduler_step_size,
                scheduler_decay=args.scheduler_decay,
                clip_gradient=args.clip_gradient,
                val_period=args.validation_period,
                optimizer=args.optimizer,
                optimizer_kwargs={
                    'lr': args.lr,
                    **({'weight_decay': args.weight_decay} if args.weight_decay is not None else {}),
                },
                early_stopping=args.early_stopping,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                model_kwargs={
                    'step_size': args.dataset_time_step,
                    'data_keys': args.data_keys,
                },
                seed=args.seed,
                device=device,
            )

        case _:  # Use default settings if none were specified
            model, _ = model_cls.fit(
                target_key=target,
                dataset=dataset,
                model_name=model_name,
            )

    return model


def _map_kwargs_adaboost(dataset_key: str) -> dict:
    # Obtained after hyperparameter search
    match dataset_key:
        case "GMU_Cherry_Japan":
            return {'n_estimators': 179}
        case "GMU_Cherry_Switzerland":
            return {'n_estimators': 130}
        case "GMU_Cherry_South_Korea":
            return {'n_estimators': 80}
        case "PEP725_Apple":
            return {'n_estimators': 138}
        case "PEP725_Pear":
            return {'n_estimators': 140}
        case "PEP725_Peach":
            return {'n_estimators': 26}
        case "PEP725_Almond":
            return {'n_estimators': 80}
        case "PEP725_Hazel":
            return {'n_estimators': 21}
        case "PEP725_Cherry":
            return {'n_estimators': 38}
        case "PEP725_Apricot":
            return {'n_estimators': 80}
        case "PEP725_Blackthorn":
            return {'n_estimators': 143}
        case "PEP725_Plum":
            return {'n_estimators': 21}
        case _:
            raise ExperimentConfigException(f'No kwargs for dataset "{dataset_key}')


def _map_kwargs_gradient_boosting(dataset_key: str) -> dict:
    # Obtained after hyperparameter search
    match dataset_key:
        case "GMU_Cherry_Japan":
            return {'n_estimators': 110, 'max_depth': 5, 'min_samples_split': 18}
        case "GMU_Cherry_Switzerland":
            return {'n_estimators': 130, 'max_depth': 1, 'min_samples_split': 18}
        case "GMU_Cherry_South_Korea":
            return {'n_estimators': 130, 'max_depth': 1, 'min_samples_split': 18}
        case "PEP725_Apple":
            return {'n_estimators': 58, 'max_depth': 7, 'min_samples_split': 19}
        case "PEP725_Pear":
            return {'n_estimators': 97, 'max_depth': 3, 'min_samples_split': 6}
        case "PEP725_Peach":
            return {'n_estimators': 97, 'max_depth': 3, 'min_samples_split': 6}
        case "PEP725_Almond":
            return {'n_estimators': 21, 'max_depth': 10, 'min_samples_split': 20}
        case "PEP725_Hazel":
            return {'n_estimators': 140, 'max_depth': 6, 'min_samples_split': 20}
        case "PEP725_Cherry":
            return {'n_estimators': 122, 'max_depth': 4, 'min_samples_split': 17}
        case "PEP725_Apricot":
            return {'n_estimators': 130, 'max_depth': 1, 'min_samples_split': 18}
        case "PEP725_Blackthorn":
            return {'n_estimators': 140, 'max_depth': 6, 'min_samples_split': 20}
        case "PEP725_Plum":
            return {'n_estimators': 58, 'max_depth': 7, 'min_samples_split': 19}
        case _:
            raise ExperimentConfigException(f'No kwargs for dataset "{dataset_key}')


def _map_kwargs_random_forest(dataset_key: str) -> dict:
    # Obtained after hyperparameter search
    match dataset_key:
        case "GMU_Cherry_Japan":
            return {'n_estimators': 132, 'max_depth': 9, 'min_samples_split': 9}
        case "GMU_Cherry_Switzerland":
            return {'n_estimators': 138, 'max_depth': 7, 'min_samples_split': 15}
        case "GMU_Cherry_South_Korea":
            return {'n_estimators': 110, 'max_depth': 5, 'min_samples_split': 18}
        case "PEP725_Apple":
            return {'n_estimators': 132, 'max_depth': 9, 'min_samples_split': 9}
        case "PEP725_Pear":
            return {'n_estimators': 135, 'max_depth': 8, 'min_samples_split': 15}
        case "PEP725_Peach":
            return {'n_estimators': 132, 'max_depth': 9, 'min_samples_split': 9}
        case "PEP725_Almond":
            return {'n_estimators': 129, 'max_depth': 8, 'min_samples_split': 3}
        case "PEP725_Hazel":
            return {'n_estimators': 134, 'max_depth': 9, 'min_samples_split': 11}
        case "PEP725_Cherry":
            return {'n_estimators': 134, 'max_depth': 9, 'min_samples_split': 11}
        case "PEP725_Apricot":
            return {'n_estimators': 21, 'max_depth': 10, 'min_samples_split': 20}
        case "PEP725_Blackthorn":
            return {'n_estimators': 132, 'max_depth': 9, 'min_samples_split': 9}
        case "PEP725_Plum":
            return {'n_estimators': 132, 'max_depth': 9, 'min_samples_split': 9}
        case _:
            raise ExperimentConfigException(f'No kwargs for dataset "{dataset_key}')

