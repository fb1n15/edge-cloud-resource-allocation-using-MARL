# from ray.rllib.models.modelv2 import ModelV2
# from ray.rllib.models.preprocessors import get_preprocessor
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# from ray.rllib.utils.annotations import override
# from ray.rllib.utils.framework import try_import_torch
#
# from models.convolutional_model import ConvolutionalModel
#
# import numpy as np
#
# torch, nn = try_import_torch()
#
#
# class CentralisedModel(TorchModelV2, nn.Module):
#     """Model for centralised control"""
#
#     def __init__(self, obs_space, action_space, num_outputs, model_config,
#                  name):
#         TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
#                               model_config, name)
#         nn.Module.__init__(self)
#
#         cnn = ConvolutionalModel(component, action_space, num_outputs, model_config,
#                                  "CNN_" + str(i))
#
#         self.logits_layer = nn.Sequential(
#
#             in_size=self.post_fc_stack.num_outputs,
#             out_size=num_outputs,
#             activation_fn=None,
#         )
#         # Create the value branch model.
#         self.value_layer = SlimFC(
#             in_size=self.post_fc_stack.num_outputs,
#             out_size=1,
#             activation_fn=None,
#             initializer=torch_normc_initializer(0.01))
#
#     @override(ModelV2)
#     def forward(self, input_dict, hidden_state, seq_lens):
#         self._features = input_dict["obs"].float().permute(0, 3, 1, 2)
#
#         x = self.model(self._features)
#         x = self._logits(x)
#
#         logits = x.squeeze(3)
#         logits = logits.squeeze(2)
#
#         return logits, hidden_state
#
#     @override(TorchModelV2)
#     def value_function(self):
#         assert self._features is not None, "must call forward() first"
#         value = self._value_branch(self._features)
#         value = value.squeeze(3)
#         value = value.squeeze(2)
#         return value.squeeze(1)
#
#
# def _get_size(obs_space):
#     return get_preprocessor(obs_space)(obs_space).size
#
#
# def combine_layers(obs_spaces):
#     """Stack layers on top of each other"""
#     return np.stack([layer for agent in obs_spaces for layer in agent])
