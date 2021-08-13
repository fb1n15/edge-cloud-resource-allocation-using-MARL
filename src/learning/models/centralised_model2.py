from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class CentralisedModelFC(TorchModelV2, nn.Module):
    """Model for centralised control"""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        local_obs_len = 37
        n_agents = obs_space.shape[0] // local_obs_len
        obs_size = _get_size(obs_space)

        print(f"number of agents = {n_agents}")
        print(f"observation size = {obs_size}")
        print(f"number of outputs = {num_outputs}")
        print(f"action_space = {action_space}")
        print(f"model_config = {model_config}")
        print(f"name = {name}")

        model_outputs = obs_size
        layers = []

        print(f"input nodes number = {model_outputs}")
        for layer_dim in model_config["custom_model_config"]["layers"]:
            layers.append(nn.Sequential(
                nn.Linear(in_features=model_outputs, out_features=layer_dim),
                nn.ReLU(),
                ))
            model_outputs = layer_dim
            print(f"this hidden layer nodes number = {model_outputs}")

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
        self._features = input_dict['obs_flat']

        print(f'input_dict["obs"] length = {len(input_dict["obs"])}')
        print(f'input_dict["obs"], one obs shape = {input_dict["obs"][0].shape}')
        print(input_dict["obs"][0])
        print(f'input_dict["obs_flat"], shape = {input_dict["obs_flat"].shape}')

        # self._features = input_dict["obs"].float().permute(0, 3, 1, 2)

        print(f"length of self._features = {len(self._features)}")
        print(self._features)

        # obs_flat = input_dict["obs_flat"].float()
        #
        # print(f"length of obs_flat = {len(obs_flat)}")
        self._features = self.model(self._features)

        print(f"length of self._features (second) = {len(self._features)}")
        x = self._logits(self._features)
        # Adding a Dimension to a Tensor in PyTorch
        # https://sparrow.dev/adding-a-dimension-to-a-tensor-in-pytorch/
        x = x.unsqueeze(0)
        print(f"x = {x}")

        # print(f"hidden_state's dim1 = {len(hidden_state)}")
        # print(f"hidden_state's dim2 = {len(hidden_state[0])}")

        return x.squeeze(1), hidden_state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        value = self._value_branch(self._features)
        value = value.unsqueeze(0)
        print(f"value = {value}")

        return value.squeeze(1)


def _get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size


def combine_layers(obs_spaces):
    """Stack layers on top of each other"""
    return torch.stack([layer for agent in obs_spaces for layer in agent], axis=1)
