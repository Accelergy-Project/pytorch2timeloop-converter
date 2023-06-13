"""
Definitions of forward hooks for various PyTorch layer types, to extract
`LayerDescription`s from them during evaluation.

For many layer types, such as 2D convolutions and self-attention
mechanisms, the layer itself does not know all of the information needed
to generate a Timeloop workload: for example, a convolutional layer does
not explicitly define its input size. As a result, we need to extract
this information while _evaluating_ the model.

The easiest mechanism for doing this is a PyTorch forward hook. This
file defines hooks for various layer types, with a primary interface
consisting of the function `hook_for()`, which returns a hook for the
given layer.

To add support for a new layer type, add a new hook type and return it
from hook_for() with the appropriate conditions.  You may also need to
add a new `LayerDescription` if the layer is very different from the
ones that are already here.
"""

from functools import singledispatch
import logging
from typing import Optional, Callable, Any
import operator

import torch
import torch.nn as nn
import transformers.models.distilbert.modeling_distilbert
from torch.fx import symbolic_trace

from pytorch2timeloop.utils.layer_descriptions import (
    DepthWiseConvLayerDescription,
    ConvLayerDescription,
    MaxPoolLayerDescription,
    MatrixMatrixMultiplyLayerDescription,
    MatmulFuncDescription
)

logger = logging.getLogger(__name__)


@singledispatch
def generate_description(module,
                         input: torch.Tensor,
                         output: torch.Tensor,
                         name: str,
                         ifmap_name: str):
    raise NotImplementedError(f'not implemented for {type(module)}')


@generate_description.register(nn.Conv2d)
def _(module, input, output, name, ifmap_name):
    if module.groups > 1 and module.groups == module.in_channels:
        description = DepthWiseConvLayerDescription(
            w=input.shape[3],
            h=input.shape[2],
            c=module.in_channels,
            s=module.kernel_size[1],
            r=module.kernel_size[0],
            w_stride=module.stride[1],
            h_stride=module.stride[0],
            w_pad=module.padding[1],
            h_pad=module.padding[0],
            n=input.shape[0],
            name=name,
            ifmap_name=ifmap_name,
            filter_name=f'{name}.filter',
            ofmap_name=f'{name}.out'
        )
    else:
        description = ConvLayerDescription(
            w=input.shape[2],
            h=input.shape[3],
            c=module.in_channels,
            m=module.out_channels,
            s=module.kernel_size[0],
            r=module.kernel_size[1],
            w_stride=module.stride[0],
            h_stride=module.stride[1],
            w_pad=module.padding[0],
            h_pad=module.padding[1],
            n=input.shape[0],
            name=name,
            ifmap_name=ifmap_name,
            filter_name=f'{name}.filter',
            ofmap_name=f'{name}.out'
        )
    return description


@generate_description.register(nn.MaxPool2d)
def _(module, input, output, name, ifmap_name):
    if isinstance(module.kernel_size, int):
        kernel_size = (module.kernel_size, module.kernel_size)
    else:
        kernel_size = module.kernel_size
    if isinstance(module.stride, int):
        stride = (module.stride, module.stride)
    else:
        stride = module.stride
    if isinstance(module.padding, int):
        padding = (module.padding, module.padding)
    else:
        padding = module.padding

    description = MaxPoolLayerDescription(
        w=input.shape[3],
        h=input.shape[2],
        c=input.shape[1],
        s=kernel_size[1],
        r=kernel_size[0],
        w_stride=stride[1],
        h_stride=stride[0],
        w_pad=padding[1],
        h_pad=padding[0],
        n=input.shape[0],
        name=name,
        ifmap_name=ifmap_name,
        ofmap_name=f'{name}.out'
    )

    return description


@generate_description.register(nn.Linear)
def _(module, input, output, name, ifmap_name):
    description = ConvLayerDescription(
        w=1,
        h=1,
        c=module.in_features,
        m=module.out_features,
        s=1,
        r=1,
        w_stride=1,
        h_stride=1,
        w_pad=0,
        h_pad=0,
        n=input.shape[0],
        name=name,
        ifmap_name=ifmap_name,
        filter_name=f'{name}.filter',
        ofmap_name=f'{name}.out'
    )
    return description


def generate_matmul_func(input1, input2, output,
                         name, input1_name, input2_name):
    if len(input1.shape) == 2 and len(input2.shape) == 2:
        description = MatmulFuncDescription(
            name = name,
            m = input1.shape[0],
            n = input2.shape[1],
            k = input1.shape[1],
            ifmap1_name = input1_name,
            ifmap2_name = input2_name,
            ofmap_name = f'{name}.out',
            extra_dims = tuple()
        )
    elif len(input1.shape) > 2 and input1.shape[:-2] == input2.shape[:-2]:
        description = MatmulFuncDescription(
            name = name,
            m = input1.shape[0],
            n = input2.shape[1],
            k = input1.shape[1],
            ifmap1_name = input1_name,
            ifmap2_name = input2_name,
            ofmap_name = f'{name}.out',
            extra_dims = input1.shape[:-2]
        )
    else:
        raise NotImplementedError(
            f'unimplemented for arg shapes {input1.shape}, {input2.shape}'
        )

    return description