from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


# https://github.com/ray-project/ray/tree/master/rllib/examples/models
# examples of customised models


class CentralisedCriticFCModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized value function.

    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).

    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A value model that also looks at the 'opponent_obs' / 'opponent_action'
      to compute the value (it does this by using the 'obs_flat' tensor).

    source: https://github.com/ray-project/ray/blob/master/rllib/examples/models
    /centralized_critic_models.py
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, n_agents):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.action_model = TorchFC(obs_space, action_space, num_outputs, model_config,
                                    name + "_action")
        # How to write the global observation space?
        self.value_model = TorchFC(obs_space * n_agents, action_space, 1, model_config,
                                   name + "_vf")

        self._model_in = None

    def forward(self, input_dict, state, seq_lens):
        # call the model with the given input tensors and state
        # Store model-input for possible `value_function()` call.
        # "obs_flat": global observation (flattened)
        self._model_in = [input_dict["obs_flat"], state, seq_lens]
        # ["obs"]["own_obs"]: local observation?
        return self.action_model({
            "obs": input_dict["obs"]["own_obs"]
            }, state, seq_lens)

    def value_function(self):
        # return the value function output for the most recent forward pass
        value_out, _ = self.value_model({
            "obs": self._model_in[0]
            }, self._model_in[1], self._model_in[2])
        return torch.reshape(value_out, [-1])
