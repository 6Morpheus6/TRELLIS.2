from .. import config
import importlib
import torch
import torch.nn as nn
from .. import SparseTensor


_backends = {}


def _remap_flex_gemm_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    """
    Remap flex_gemm-format checkpoint keys to the current backend's format.
    flex_gemm stores weights directly as `weight`/`bias`, while spconv and
    torchsparse store them inside a `conv` submodule (`conv.weight`/`conv.bias`).
    """
    if config.CONV == 'flex_gemm':
        return
    # Check if this state_dict has flex_gemm-format keys (weight directly, not conv.weight)
    weight_key = f'{prefix}weight'
    bias_key = f'{prefix}bias'
    conv_weight_key = f'{prefix}conv.weight'
    if weight_key in state_dict and conv_weight_key not in state_dict:
        state_dict[conv_weight_key] = state_dict.pop(weight_key)
        if bias_key in state_dict:
            state_dict[f'{prefix}conv.bias'] = state_dict.pop(bias_key)


class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        if config.CONV not in _backends:
            _backends[config.CONV] = importlib.import_module(f'..conv_{config.CONV}', __name__)
        _backends[config.CONV].sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride, dilation, padding, bias, indice_key)
        if config.CONV != 'flex_gemm':
            self._register_load_state_dict_pre_hook(_remap_flex_gemm_state_dict)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return _backends[config.CONV].sparse_conv3d_forward(self, x)


class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super(SparseInverseConv3d, self).__init__()
        if config.CONV not in _backends:
            _backends[config.CONV] = importlib.import_module(f'..conv_{config.CONV}', __name__)
        _backends[config.CONV].sparse_inverse_conv3d_init(self, in_channels, out_channels, kernel_size, stride, dilation, bias, indice_key)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return _backends[config.CONV].sparse_inverse_conv3d_forward(self, x)
