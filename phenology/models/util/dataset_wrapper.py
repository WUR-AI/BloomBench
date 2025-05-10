from collections import defaultdict

import torch
import torch.utils.data

import phenology.config as config
from phenology.dataset.dataset import Dataset
from phenology.util.func_torch import batch_tensors


class TorchDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __getitem__(self, index) -> dict:
        """
        Get a sample from the dataset
        Data is cast to PyTorch tensors where required
        :param index: index that is passed to the dataset
        :return: a dict containing the data sample
        """
        return self.cast_to_tensor(
            self._dataset[index],
        )

    def __len__(self):
        return len(self._dataset)

    @classmethod
    def cast_to_tensor(cls, item: dict) -> dict:
        """
        Create a data item with all data cast to torch tensors
        :param item: the data item to convert
        :return: the converted data item
        """

        ys = item[config.KEY_OBSERVATIONS_INDEX]
        fs = item[config.KEY_FEATURES]

        ys = {
            k: torch.tensor(v, dtype=torch.float) for k, v in ys.items()
        }
        fs = {
            k: torch.tensor(v, dtype=torch.float) for k, v in fs.items()
        }

        item_copy = {
            k: v for k, v in item.items() if k not in [config.KEY_FEATURES, config.KEY_OBSERVATIONS_INDEX]
        }

        item_copy[config.KEY_FEATURES] = fs
        item_copy[config.KEY_OBSERVATIONS_INDEX] = ys

        return item_copy

    # @classmethod
    # def collate_fn(cls, items: list) -> dict:
    #     """
    #     Function that takes a list of data items (as dicts, containing torch tensors) and converts it to a dict of
    #     batched torch tensors
    #     :param items: a list of data items
    #     :return: a dict with batched data
    #     """
    #     assert len(items) > 0
    #
    #     batch = defaultdict(list)
    #
    #     for item in items:
    #         for k, v in item.items():
    #             batch[k].append(v)
    #
    #     # Ensure all keys have the same nr. of entries
    #     # assert len(set([len(vs) for vs in batch.values()])) == 1
    #
    #     # Two keys now map to lists of dicts containing tensors, namely
    #     # - The observation types -> observed value
    #     # - The feature types -> observed feature values
    #     # Both lists require conversion to a dict of batched tensors
    #     # First step is to convert both to dicts containing lists of tensors
    #     batched_obs_ix = defaultdict(list)
    #     for obs_ix in batch[config.KEY_OBSERVATIONS_INDEX]:
    #         for k, v in obs_ix.items():
    #             batched_obs_ix[k].append(v)
    #     batched_features = defaultdict(list)
    #     for features in batch[config.KEY_FEATURES]:
    #         for k, v in features.items():
    #             batched_features[k].append(v)
    #     # Now batch the lists
    #     batched_obs_ix = {
    #         k: batch_tensors(*v) for k, v in batched_obs_ix.items()
    #     }
    #     batched_features = {
    #         k: batch_tensors(*v) for k, v in batched_features.items()
    #     }
    #     # Overwrite the batched tensors
    #     batch = dict(batch)  # Cast to normal dict since we no longer expect them to contain lists
    #     batch[config.KEY_OBSERVATIONS_INDEX] = batched_obs_ix
    #     batch[config.KEY_FEATURES] = batched_features
    #
    #     return batch

    @classmethod
    def collate_fn(cls, items: list) -> dict:
        """
        Function that takes a list of data items (as dicts, containing torch tensors) and converts it to a dict of
        batched torch tensors
        :param items: a list of data items
        :return: a dict with batched data
        """
        assert len(items) > 0

        batch = defaultdict(list)

        for item in items:
            for k, v in item.items():
                if k != config.KEY_OBSERVATIONS_INDEX and k != config.KEY_FEATURES:
                    batch[k].append(v)  # TODO -- KEY_OBSERVATION is not handled properly
        batch = dict(batch)

        batch[config.KEY_OBSERVATIONS_INDEX] = dict()
        batch[config.KEY_FEATURES] = dict()

        for k in items[0][config.KEY_OBSERVATIONS_INDEX].keys():
            batch[config.KEY_OBSERVATIONS_INDEX][k] = batch_tensors(
                *[item[config.KEY_OBSERVATIONS_INDEX][k] for item in items]
            )

        for k in items[0][config.KEY_FEATURES].keys():
            batch[config.KEY_FEATURES][k] = batch_tensors(
                *[item[config.KEY_FEATURES][k] for item in items]
            )

        return batch

    @staticmethod
    def sample_to_device(sample: dict, device) -> dict:
        """
        Function that takes a sample and moves tensors it contains to the specified device (in-place)
        :param sample: the sample to convert
        :param device: the device to move tensors to
        :return: the sample but moved to the specified device
        """

        moved_sample = {
            k: v for k, v in sample.items() if k != config.KEY_FEATURES and k != config.KEY_OBSERVATIONS_INDEX
        }

        moved_sample[config.KEY_FEATURES] = {
            k: v.to(device) for k, v in sample[config.KEY_FEATURES].items()
        }

        moved_sample[config.KEY_OBSERVATIONS_INDEX] = {
            k: v.to(device) for k, v in sample[config.KEY_OBSERVATIONS_INDEX].items()
        }

        return moved_sample

    @staticmethod
    def get_default_norm_params() -> dict:
        return {
            # Hourly
            'temperature_2m': (8.431772, 7.707247),
            'is_day': (0.3705145, 0.4829339),
            'precipitation': (0.09331062, 0.29683942),
            'relative_humidity_2m': (78.96578, 14.400801),
            'soil_moisture_0_to_7cm': (0.35431248, 0.07339167),
            'soil_moisture_7_to_28cm': (0.3520766, 0.07325297),
            'shortwave_radiation': (121.193474, 190.0891),

            # Daily
            'temperature_2m_max': (11.510296, 7.7762413),
            'temperature_2m_min': (4.510901, 6.667528),
            'temperature_2m_mean': (8.022747, 7.1008506),
            'daylight_duration': (42952.715, 10155.233),  # In seconds
        }


if __name__ == '__main__':

    from tqdm import tqdm

    with Dataset.load(
        # key='debug',
        key='CPF_PEP725_winter_wheat',
    ) as _dataset:

        for _ in tqdm(_dataset.iter_items(), desc='1st iteration normal', total=len(_dataset)):
            pass

        for _ in tqdm(_dataset.iter_items(), desc='2nd iteration normal', total=len(_dataset)):
            pass

        _dataset = TorchDataset(_dataset)

        _dataloader = torch.utils.data.DataLoader(_dataset,
                                                  batch_size=2,
                                                  collate_fn=TorchDataset.collate_fn,
                                                  shuffle=True,
                                                  )

        # print(_dataset[0])

        for _batch in tqdm(_dataloader, desc='1st iteration wrapped', total=len(_dataloader)):
            # print(_batch)
            pass

        for _batch in tqdm(_dataloader, desc='2nd iteration wrapped', total=len(_dataloader)):
            # print(_batch)
            pass
