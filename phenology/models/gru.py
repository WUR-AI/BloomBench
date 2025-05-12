import torch
import torch.nn as nn
import torch.nn.functional as F

import phenology.config as config
from phenology.models.base_torch import BaseTorchModel
from phenology.models.util.dataset_wrapper import TorchDataset


class GRUModel(BaseTorchModel):

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

        self._rnn = nn.GRU(input_size=num_features,
                           hidden_size=self._hidden_size,
                           batch_first=True,
                           num_layers=num_layers,
                           )

        self._lin = nn.Conv1d(in_channels=self._hidden_size,
                              out_channels=1,
                              kernel_size=1,
                              )

    def forward(self, xs: dict, **kwargs) -> tuple:

        xs = self.normalize_sample(xs, self._feature_statistics)

        # concatenate feature data
        #   each feature is of shape (batch_size, season_length)
        #   create tensor of shape (batch_size, season_length, num_features)
        fs = [xs[config.KEY_FEATURES][key] for key in self._data_keys]
        fs = torch.cat([f.unsqueeze(-1) for f in fs], dim=-1)

        fs = torch.nan_to_num(fs)

        # pass feature data through the GRU
        #   resulting tensor is of shape (batch_size, season_length, hidden_size)
        fs, _ = self._rnn(fs)

        # set hidden state as channel dimension
        fs = fs.permute(0, 2, 1)

        # apply final linear layer
        #   resulting tensor is of shape (batch_size, 1 (channel), season_length)
        fs = F.sigmoid(self._lin(fs))

        # remove channel dimension
        fs = fs.squeeze(1)

        # set indices to be the days with max change in activation (max probability of the event occurring that day)
        ixs = torch.argmax(fs - torch.roll(fs, 1, dims=-1), dim=-1)
        # clamp values between 0 and season_length - 1 to be valid indices
        ixs = ixs.clamp(min=0, max=fs.size(-1) - 1)

        return ixs, {
            'ps': fs,
        }

    def loss(self, xs: dict, target_key: str, scale: float = 1) -> tuple:  # TODO -- make compatible with hourly target ix
        ys_pred, info = self(xs)

        ps = info['ps']
        bs = ps.size(0)
        sl = ps.size(-1)

        ys_true = xs[config.KEY_OBSERVATIONS_INDEX][target_key]

        ps_true = torch.cat(
            [torch.arange(sl).unsqueeze(0) for _ in range(bs)], dim=0
        ).to(ps.device)

        ps_true = (ps_true >= ys_true.view(-1, 1)).to(ps.dtype)

        loss = F.binary_cross_entropy(ps, ps_true)

        return loss, {
            'forward_pass': info,
            'ys_pred': ys_pred,
            'ys_true': ys_true,
        }

