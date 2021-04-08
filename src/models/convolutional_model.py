from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class ConvolutionalModel(TorchModelV2, nn.Module):
    """Model for an image style obsevation space"""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        self.obs_size = _get_size(obs_space)
        self.rnn_hidden_dim = model_config["lstm_cell_size"]
        self.cnn_outputs = 512
        self.model = nn.Sequential(
            nn.ZeroPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),

            nn.ZeroPad2d(padding=(0, 1, 0, 1)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=self.cnn_outputs, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self._value_branch = nn.Sequential(
            nn.ZeroPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),

            nn.ZeroPad2d(padding=(0, 1, 0, 1)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=self.cnn_outputs, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=self.cnn_outputs, out_channels=1, kernel_size=1, stride=1),
        )

        self._logits = nn.Sequential(
            nn.ZeroPad2d(padding=(0, 0, 0, 0)),
            nn.Conv2d(self.cnn_outputs, num_outputs, 1, 1)
        )

    @override(ModelV2)
    def forward(self, input_dict, hidden_state, seq_lens):
        self._features = input_dict["obs"].float().permute(0, 3, 1, 2)

        x = self.model(self._features)
        x = self._logits(x)

        logits = x.squeeze(3)
        logits = logits.squeeze(2)

        return logits, hidden_state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        value = self._value_branch(self._features)
        value = value.squeeze(3)
        value = value.squeeze(2)
        return value.squeeze(1)


def _get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size
