import logging
import math
import operator
from typing import Dict, Tuple, Union

import torch
from torch import nn
import torch.fx as fx

from .converter import generate_description, generate_matmul_func
from pytorch2timeloop.utils.layer_descriptions import BinaryElementwiseFuncDescription

logger = logging.getLogger(__name__)


class Converter(fx.Interpreter):
    DEFAULT_BYPASSED_MODULES = Union[
        nn.BatchNorm2d,
        nn.Dropout,
        # Elementwise activations
        nn.Hardsigmoid,
        nn.Hardswish,
        nn.ReLU,
        nn.ReLU6
    ]

    DEFAULT_IGNORED_MODULES = Union[
        nn.AdaptiveAvgPool2d
    ]

    UNARY_ELEMENTWISE_FUNC = [
        math.sqrt
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

    def __init__(self, module, garbage_collect_values=True,
                    bypassed_modules=None, ignored_modules=None):
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
            self.bypassed_arg_remap[f'{name}.out'] = \
                f'{original_args[0].name}.out'
            return result

        arg_name = original_args[0].name
        while arg_name in self.bypassed_arg_remap:
            arg_name = self.bypassed_arg_remap[arg_name]

        description = generate_description(module, args[0], result, name,
                                           original_args[0].name)

        self.summary.append(description)

        return result
    
    def call_function(self, target, args, kwargs, name: str,
                      original_args: tuple):
        result = super().call_function(target, args, kwargs)

        if target in Converter.BINARY_ELEMENTWISE_FUNC:
            if isinstance(args[1], torch.Tensor):
                description = BinaryElementwiseFuncDescription(
                    ifmap1_shape = args[0].shape,
                    ifmap2_shape = args[1].shape,
                    ofmap_shape = result.shape,
                    ifmap1_name = f'{original_args[0].name}.out',
                    ifmap2_name = f'{original_args[1].name}.out',
                    ofmap_name = f'{name}.out',
                    name = name
                )
                self.summary.append(description)
            logger.warning('assuming op by scalar %s[type=%s] args=%s',
                           name, target, args)
        elif target == torch.matmul:
            description = generate_matmul_func(
                input1=args[0],
                input2=args[1],
                output=result,
                name=name,
                input1_name=f'{original_args[0].name}.out',
                input2_name=f'{original_args[1].name}.out'
            )
        elif target == nn.softmax:
            raise NotImplementedError('softmax unimplemented')
        elif target in Converter.UNARY_ELEMENTWISE_FUNC:
            pass
        else:
            logger.error('unknwown function  %s[type=%s]', name, target)
            raise NotImplementedError()

        return result