import torch
import torch.nn as nn
import torch.nn.functional as F

import phenology.config as config
from phenology.models.base_torch import BaseTorchModel
from phenology.models.util.dataset_wrapper import TorchDataset


class CNNModel(BaseTorchModel):

    def __init__(self,
                 step_size: str,
                 data_keys: str,
                 feature_statistics: dict = None,
                 hidden_size: int = 16,
                 num_layers: int = 3,
                 ):
        super().__init__()

        # set normalization parameters
        if feature_statistics is None:
            self._feature_statistics = TorchDataset.get_default_norm_params()
        else:
            self._feature_statistics = feature_statistics

        self._step_size = step_size
        self._data_keys = data_keys

        self._hidden_size = hidden_size

        num_features = len(self._data_keys)

        l_in = nn.Conv1d(in_channels=num_features,
                         out_channels=hidden_size,
                         kernel_size=3,
                         padding=1,
                         )
        last_pool = nn.AdaptiveAvgPool1d(64)
        l_out = nn.Linear(64, 1)

        self._cnn = nn.ModuleList([l_in] + [
            nn.ModuleList([
                nn.AvgPool1d(kernel_size=2,),
                nn.Conv1d(in_channels=hidden_size,
                          out_channels=hidden_size,
                          kernel_size=3,
                          padding=1,
                          )
            ])
            for _ in range(num_layers - 1)
        ] + [last_pool, nn.Flatten(), l_out])

    def forward(self, xs: dict, **kwargs) -> tuple:

        xs = self.normalize_sample(xs, self._feature_statistics)

        # concatenate feature data
        #   each feature is of shape (batch_size, season_length)
        #   create tensor of shape (batch_size, num_features (channels), season_length)
        fs = [xs[config.KEY_FEATURES][key] for key in self._data_keys]
        fs = torch.cat([f.unsqueeze(1) for f in fs], dim=1)

        fs = torch.nan_to_num(fs)

        ixs = self._cnn(fs).view(-1)

        # clamp values between 0 and season_length - 1 to be valid indices
        ixs = ixs.clamp(min=0, max=fs.size(-1) - 1)

        return ixs, {}
