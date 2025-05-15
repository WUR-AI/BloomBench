import numpy as np

from phenology.evaluation.evaluation import EvaluationResult
from phenology.models.adaboost import AdaBoostModel
from phenology.models.cnn import CNNModel
from phenology.models.gradient_boosting import GradientBoostingModel
from phenology.models.gru import GRUModel
from phenology.models.linear import LinearTrendModel
from phenology.models.lstm import LSTMModel
from phenology.models.mean import MeanModel
from phenology.models.random_forest import RandomForestModel

if __name__ == '__main__':

    dataset_keys = (  # In same order as the paper!
        'PEP725_Apple',
        'PEP725_Pear',
        'PEP725_Peach',
        'PEP725_Almond',
        'PEP725_Hazel',
        'PEP725_Apricot',
        'PEP725_Plum',
        'PEP725_Blackthorn',
        'PEP725_Cherry',
        'GMU_Cherry_Japan',
        'GMU_Cherry_Switzerland',
        'GMU_Cherry_South_Korea',
    )

    model_keys = (
        MeanModel.__name__,
        LinearTrendModel.__name__,
        RandomForestModel.__name__,
        GradientBoostingModel.__name__,
        AdaBoostModel.__name__,
        CNNModel.__name__,
        GRUModel.__name__,
        LSTMModel.__name__,
    )

    seeds = (
        94, 85, 30, 93, 50,
    )

    run_prefix = 'fit_eval'

    for dataset_key in dataset_keys:

        maes_avg_trn = []
        maes_avg_tst = []

        maes_std_trn = []
        maes_std_tst = []

        for model_key in model_keys:
            maes_trn = []
            maes_tst = []

            for seed in seeds:
                model_name = f'{model_key}_{dataset_key}_seed_{seed}'
                run_name = f'{run_prefix}_{model_name}'

                eval_result = EvaluationResult.load(run_name)
                metrics = eval_result.compute_metrics()

                metrics_trn = metrics['train']
                metrics_tst = metrics['test']

                mae_trn = metrics_trn.loc['mae'].iloc[0]
                mae_tst = metrics_tst.loc['mae'].iloc[0]

                maes_trn.append(mae_trn)
                maes_tst.append(mae_tst)

            mae_avg_trn = np.mean(maes_trn)
            mae_avg_tst = np.mean(maes_tst)

            mae_std_trn = np.std(maes_trn)
            mae_std_tst = np.std(maes_tst)

            maes_avg_trn.append(mae_avg_trn)
            maes_avg_tst.append(mae_avg_tst)

            maes_std_trn.append(mae_std_trn)
            maes_std_tst.append(mae_std_tst)

        line_trn = ' & '.join(
            [dataset_key.replace('_', '\\_')] +
            [f'${mae_avg:.2f} \\pm {mae_std:.2f}$' for mae_avg, mae_std in zip(maes_avg_trn, maes_std_trn)]
        ) + '\\\\'
        line_tst = ' & '.join(
            [dataset_key.replace('_', '\\_')] +
            [f'${mae_avg:.2f} \\pm {mae_std:.2f}$' for mae_avg, mae_std in zip(maes_avg_tst, maes_std_tst)]
        ) + '\\\\'

        print(line_trn)
        print()
        print(line_tst)

