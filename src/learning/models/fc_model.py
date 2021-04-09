from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class FCModel(TorchModelV2, nn.Module):
    """"""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        # Calculate flattened obs size
        obs_size = _get_size(obs_space)

        model_outputs = obs_size

        layers = []

        for layer_dim in model_config["custom_model_config"]["layers"]:
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
