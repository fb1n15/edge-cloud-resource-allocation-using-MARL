from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class FCModel(TorchModelV2, nn.Module):
    """
    If you would like to provide your own model logic, you can sub-class either TFModelV2 (for TensorFlow) or TorchModelV2 (for PyTorch)
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
            name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
            model_config, name)
        nn.Module.__init__(self)

        # Calculate flattened obs size
        obs_size = _get_size(obs_space)


        layers = []

        prev_layer_dim = obs_size
        """create layers (0 to second-last)
        Args:
            in_features: size of the input sample
            out_features: size of the output sample
        """
        for layer_dim in model_config["custom_model_config"]["layers"]:
            layers.append(nn.Sequential(
                nn.Linear(in_features=prev_layer_dim, out_features=layer_dim),
                nn.ReLU(),
            ))
            prev_layer_dim = layer_dim

        self.model = nn.Sequential(
            *layers
        )

        self._value_branch = nn.Sequential(
            nn.Linear(in_features=prev_layer_dim, out_features=1)
        )

        self._logits = nn.Sequential(
            nn.Linear(in_features=prev_layer_dim, out_features=num_outputs)
        )

    @override(ModelV2)
    def forward(self, input_dict, hidden_state, seq_lens):
        """ This is the Actor network
        forward() takes a dict of tensor inputs (mapping str to Tensor types), whose keys and values depend on the view requirements of the model. Normally, this input dict contains only the current observation obs and an is_training boolean flag, as well as an optional list of RNN states.
        Return:
            output tensor, the new state
        """
        obs_flat = input_dict["obs_flat"].float()

        self._features = self.model(obs_flat)
        x = self._logits(self._features)

        return x.squeeze(1), hidden_state

    @override(TorchModelV2)
    def value_function(self):
        """ This is the Critic network?
        Returns the value function output for the most recent forward pass
        """
        assert self._features is not None, "must call forward() first"
        value = self._value_branch(self._features).squeeze(1)

        return value


def _get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size
