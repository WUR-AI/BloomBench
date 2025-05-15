from phenology.dataset.dataset import Dataset

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

        print(dataset_key)
        print(f'Nr. of observations: {len(dataset)}')
        print()