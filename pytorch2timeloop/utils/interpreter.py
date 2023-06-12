import logging
import operator
from typing import Dict, Tuple, Union

import torch
from torch import nn
import torch.fx as fx

from .converter import generate_description, generate_func_description

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

        if target == operator.add or target == torch.add:
            description = generate_func_description(
                target,
                args[0],
                result,
                name,
                original_args[0].name
            )
            self.summary.append(description)

        return result