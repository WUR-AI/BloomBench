import os.path

from phenology.dataset.base import BaseDataset

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

    dpi = 500

    # Use same dir as script to store figures
    path_base = os.path.abspath(os.path.dirname(__file__))

    for key in dataset_keys:
        print(key)

        dataset = BaseDataset.load(key)

        path = os.path.join(
            path_base,
            'figures',
            'datasets',
            key,
        )

        match key:
            case 'GMU_Cherry_Japan':
                obs_type = 'gmu_0'
            case 'GMU_Cherry_Switzerland':
                obs_type = 'gmu_1'
            case 'GMU_Cherry_South_Korea':
                obs_type = 'gmu_2'
            case 'PEP725_Apple':
                obs_type = 'BBCH_60'
            case 'PEP725_Pear':
                obs_type = 'BBCH_60'
            case 'PEP725_Peach':
                obs_type = 'BBCH_60'
            case 'PEP725_Almond':
                obs_type = 'BBCH_60'
            case 'PEP725_Hazel':
                obs_type = 'BBCH_60'
            case 'PEP725_Cherry':
                obs_type = 'BBCH_60'
            case 'PEP725_Apricot':
                obs_type = 'BBCH_60'
            case 'PEP725_Blackthorn':
                obs_type = 'BBCH_60'
            case 'PEP725_Plum':
                obs_type = 'BBCH_60'
            # case 'PEP725_Oak':
            #     obs_type = 'BBCH_11'

            case _:
                raise KeyError(f'Key {key} has no matching obs type for plotting a linear trend')

        dataset.savefig_observation_mean_over_time(os.path.join(path, 'observation_trend'),
                                                   obs_type=obs_type,
                                                   dpi=dpi,
                                                   )

        dataset.savefig_observation_map(
            path=os.path.join(path, 'observation_map'),
        )

        # self.savefigs_observation_maps_over_time(
        #     path=os.path.join(path, 'observation_maps_over_time'),
        #     verbose=False,
        # )

        dataset.savefig_observation_hists(
            path=os.path.join(path, 'observation_hists'),
        )

        dataset.savefig_species_subgroup_occurrence_temporal(
            path=os.path.join(path, 'species_subgroup_occurrence_temporal'),
        )

