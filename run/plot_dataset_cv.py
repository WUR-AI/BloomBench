import os
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from phenology import config
from phenology.dataset.dataset import Dataset
from phenology.util.func import round_partial

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


if __name__ == '__main__':

    dataset_keys = [
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

    for dataset_key in dataset_keys:

        dataset = Dataset.load(dataset_key)

        cells_to_samples = defaultdict(list)

        observations = []

        # res_lat, res_lon = 0.25, 0.25
        res_lat, res_lon = 0.5, 0.5
        # res_lat, res_lon = 1.5, 1.5
        # res_lat, res_lon = 2.5, 2.5

        target_key = DATASET_KEY_TO_TARGET[dataset_key]

        for sample in dataset.iter_items():

            src_key = sample[config.KEY_DATA_SOURCE]
            loc_id = sample[config.KEY_LOC_ID]
            year = sample[config.KEY_YEAR]
            species = sample[config.KEY_SPECIES_CODE]
            subgroup = sample[config.KEY_SUBGROUP_CODE]

            coords = dataset.get_location_coordinates(i=(src_key, loc_id), from_index=False)
            lat = coords[config.KEY_LAT]
            lon = coords[config.KEY_LON]

            lat = round_partial(lat, res_lat)
            lon = round_partial(lon, res_lon)

            cells_to_samples[(lat, lon), year].append(sample)
            observations.append(sample[config.KEY_OBSERVATIONS_INDEX][target_key])

        obs_mean = np.mean(observations)

        # Compute cell statistics
        cell_means = {cell: np.mean([ob[config.KEY_OBSERVATIONS_INDEX][target_key] for ob in obs]) for cell, obs in cells_to_samples.items()}
        cell_stds = {cell: np.std([ob[config.KEY_OBSERVATIONS_INDEX][target_key] for ob in obs]) for cell, obs in cells_to_samples.items()}
        cell_counts = {cell: len(obs) for cell, obs in cells_to_samples.items()}

        cells = list(cells_to_samples.keys())
        mean_cv = np.mean([cell_stds[c] / cell_means[c] for c in cells if cell_counts[c] > 1]) * 100
        mean_std = np.mean([cell_stds[c] for c in cells if cell_counts[c] > 1])
        # mean_std = np.mean(list(cell_stds.values()))
        print(dataset_key)
        print(f'Nr. of observations: {len(dataset)}')
        print(f'Nr. of cells: {len(cells)}')
        print(f'Mean CV: {mean_cv:.4f} %')
        print(f'Mean STD: {mean_std:.2f}')

        path_base = os.path.abspath(os.path.dirname(__file__))

        path = os.path.join(
            path_base,
            'figures',
            'datasets',
            dataset_key,
            'cell_observation_variation',
        )
        os.makedirs(path, exist_ok=True)

        fig, ax = plt.subplots()

        ax.scatter([cell_counts[c] for c in cells if cell_counts[c] > 1],
                   [cell_stds[c] / cell_means[c] for c in cells if cell_counts[c] > 1],
                   alpha=0.01,
                   label=f'cell size ({res_lat}, {res_lon})'
                   )

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        max_count = max([cell_counts[c] for c in cells])
        # tick_count = 8
        # tick_step = max_count // tick_count
        tick_step = 5
        xticks = list(range(0, max_count, tick_step))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(xtick) for xtick in xticks])

        ax.set_xlabel('Observation count')
        ax.set_ylabel(f'CV (%) ($\\mu={obs_mean:.1f}$)')
        # ax.set_title('Coefficient of Variation ($\\frac{\\sigma}{\\mu}$) against nr. of observations per cell per year')
        ax.set_title(f'N obs: {len(dataset)}, N cells: {len(cells)}, Mean CV {mean_cv:.4f} %, Mean std: {mean_std:.2f}')

        plt.legend()
        plt.savefig(os.path.join(path, f'cell_cv_grid_{(res_lat, res_lon)}.png'))
        # plt.show()
        plt.cla()
        plt.close()

