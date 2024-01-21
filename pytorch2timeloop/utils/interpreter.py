import logging
import math
import operator
from typing import Dict, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
import torch.fx as fx

from .converter import generate_description, generate_matmul_func
from pytorch2timeloop.utils.layer_descriptions import (
    BinaryElementwiseFuncDescription,
    SoftmaxFuncDescription,
    MaxPoolLayerDescription,
    ViewFuncDescription
)

logger = logging.getLogger(__name__)


class Converter(fx.Interpreter):
    DEFAULT_BYPASSED_MODULES = (
        nn.BatchNorm2d,
        nn.Dropout,
        # Elementwise activations
        nn.Hardsigmoid,
        nn.Hardswish,
        nn.ReLU,
        nn.ReLU6
    )

    DEFAULT_IGNORED_MODULES = tuple()

    UNARY_ELEMENTWISE_FUNC = [
        math.sqrt,
        F.relu,
        F.relu6
    ]

    BINARY_ELEMENTWISE_FUNC = [
        operator.add,
        torch.add,
        operator.sub,
        torch.sub,
        operator.mul,
        torch.mul,
        operator.truediv,
        torch.div
    ]

    DEFAULT_IGNORED_FUNC = []

    SOFTMAX = [
        torch.softmax,
        F.softmax
    ]

    def __init__(self, module, garbage_collect_values=True,
                 bypassed_modules=None, ignored_modules=None,
                 ignored_func=None):
        super().__init__(module, garbage_collect_values)
        self.name_to_module = dict(module.named_modules())
        self.tensor_sizes = {}
        self.summary = []

        if bypassed_modules is None:
            bypassed_modules = Converter.DEFAULT_BYPASSED_MODULES
        self.bypassed_modules = bypassed_modules

        if ignored_modules is None:
            ignored_modules = Converter.DEFAULT_IGNORED_MODULES
        self.ignored_modules = ignored_modules

        if ignored_func is None:
            ignored_func = Converter.DEFAULT_IGNORED_FUNC
        self.ignored_func = ignored_func

        self.bypassed_arg_remap = {}

    def run_node(self, n):
        name = n.name
        original_args = n.args
        with self._set_current_node(n):
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            if n.op == 'call_module' or n.op == 'call_function':
                return getattr(self, n.op)(n.target, args, kwargs, name,
                                           original_args)
            return getattr(self, n.op)(n.target, args, kwargs)

    def call_module(self, target, args: Tuple, kwargs: Dict, name: str,
                    original_args: tuple):
        result = super().call_module(target, args, kwargs)
        module = self.name_to_module[target]

        if isinstance(module, self.ignored_modules):
            logger.warning('ignoring module %s[type=%s]', name, module)
            return result

        if isinstance(module, self.bypassed_modules):
            self.bypassed_arg_remap[f'{name}_out'] = \
                f'{original_args[0].name}_out'
            return result

        arg_name = f'{original_args[0].name}_out'
        while arg_name in self.bypassed_arg_remap:
            arg_name = self.bypassed_arg_remap[arg_name]

        description = generate_description(module, args[0], result, name,
                                           arg_name)

        self.summary.append(description)

        return result
    
    def call_function(self, target, args, kwargs, name: str,
                      original_args: tuple):
        result = super().call_function(target, args, kwargs)

        arg_names = []
        for arg in original_args:
            try:
                arg_names.append(f'{arg.name}_out')
            except:
                arg_names.append(None)

        for i, n in enumerate(arg_names):
            if n is not None:
                while n in self.bypassed_arg_remap:
                    n = self.bypassed_arg_remap[n]
                arg_names[i] = n

        if target in self.ignored_func:
            logger.warning('ignoring func %s[type=%s]', name, target)
            pass
        elif target in Converter.BINARY_ELEMENTWISE_FUNC:
            if isinstance(args[1], torch.Tensor):
                description = BinaryElementwiseFuncDescription(
                    ifmap1_shape = args[0].shape,
                    ifmap2_shape = args[1].shape,
                    ofmap_shape = result.shape,
                    ifmap1_name = arg_names[0],
                    ifmap2_name = arg_names[1],
                    ofmap_name = f'{name}_out',
                    name = name
                )
                self.summary.append(description)
        elif target == F.adaptive_avg_pool2d:
            stride_w = args[0].shape[-1] // result.shape[-1]
            stride_h = args[0].shape[-2] // result.shape[-2]
            kernel_w = args[0].shape[-1] - (result.shape[-1]-1)*stride_w
            kernel_h = args[0].shape[-2] - (result.shape[-2]-1)*stride_h

            description = MaxPoolLayerDescription(
                w=args[0].shape[3],
                h=args[0].shape[2],
                c=args[0].shape[1],
                s=kernel_w,
                r=kernel_h,
                w_stride=stride_w,
                h_stride=stride_h,
                w_pad=0,
                h_pad=0,
                n=args[0].shape[0],
                name=name,
                ifmap_name=arg_names[0],
                ofmap_name=f'{name}_out'
            )
            self.summary.append(description)
        elif target == torch.matmul:
            description = generate_matmul_func(
                input1 = args[0],
                input2 = args[1],
                output = result,
                name = name,
                input1_name = arg_names[0],
                input2_name = arg_names[1]
            )
            self.summary.append(description)
        elif target in Converter.SOFTMAX:
            description = SoftmaxFuncDescription(
                ifmap_shape = args[0].shape,
                ofmap_shape = result.shape,
                ifmap_name = arg_names[0],
                ofmap_name = f'{name}_out',
                name = name,
                softmax_dim = kwargs['dim']
            )
            self.summary.append(description)
        elif target == torch.flatten:
            description = ViewFuncDescription(
                name=name,
                ifmap_shape=args[0].shape,
                ofmap_shape=result.shape,
                ifmap_name=arg_names[0],
                ofmap_name=f'{name}_out'
            )
            self.summary.append(description)
        elif target in Converter.UNARY_ELEMENTWISE_FUNC:
            self.bypassed_arg_remap[f'{name}.out'] = \
                f'{original_args[0].name}.out'
            pass
        else:
            logger.error('unknwown function  %s[type=%s]', name, target)
            raise NotImplementedError()

        return result
