from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class CentralisedModelFC(TorchModelV2, nn.Module):
    """Model for centralised control (not using convolutional layers?)"""
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        WIDTH, HEIGHT, DEPTH = 11, 11, 3
        n_agents = obs_space.shape[0] // (WIDTH * HEIGHT * DEPTH)
        obs_size = _get_size(obs_space)

        model_outputs = obs_size * n_agents
        layers = []

        for layer_dim in model_config["custom_model_config"]["layers"]:
        # for layer_dim in [256, 256]:
            layers.append(nn.Sequential(
            nn.Linear(in_features=model_outputs, out_features=layer_dim),
            nn.ReLU(),
            ))
            model_outputs = layer_dim

        self.model = nn.Sequential(
            *layers
        )

        self._value_branch = nn.Sequential(
            nn.Linear(in_features=model_outputs, out_features=1)
        )

        self._logits = nn.Sequential(
            nn.Linear(in_features=model_outputs, out_features=num_outputs)
        )

    @override(ModelV2)
    def forward(self, input_dict, hidden_state, seq_lens):
        self._features = combine_layers(component.float().permute(3, 0, 1, 2) for component in input_dict["obs"]).flatten()

        obs_flat = input_dict["obs_flat"].float()

        self._features = self.model(obs_flat)
        x = self._logits(self._features)

        return x.squeeze(1), hidden_state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        value = self._value_branch(self._features)

        return value.squeeze(1)


def _get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size


def combine_layers(obs_spaces):
    """Stack layers on top of each other"""
    return torch.stack([layer for agent in obs_spaces for layer in agent], axis=1)
