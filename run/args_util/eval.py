import argparse

from phenology.dataset.dataset import Dataset
from phenology.evaluation.evaluation import evaluate, EvaluationResult
from phenology.models.base import BaseModel


def evaluate_model_using_args(model: BaseModel,
                              dataset_trn: Dataset,
                              dataset_tst: Dataset,
                              args: argparse.Namespace,
                              ) -> EvaluationResult:

    run_name = args.run_name
    model_name = args.model_name
    target_key = args.target

    eval_result = evaluate(model=model,
                           dataset_train=dataset_trn,
                           dataset_test=dataset_tst,
                           target=target_key,
                           run_name=run_name,
                           model_name=model_name,
                           )

    return eval_result


