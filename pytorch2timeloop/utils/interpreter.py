from typing import Dict, Tuple

import torch
import torch.fx as fx

from .converter import generate_description

class Converter(fx.Interpreter):
    def __init__(self, module, garbage_collect_values=True):
        super().__init__(module, garbage_collect_values)
        self.name_to_module = dict(module.named_modules())
        self.tensor_sizes = {}
        self.summary = []

    def call_module(self, target, args: Tuple, kwargs: Dict):
        result = super().call_module(target, args, kwargs)
        module = self.name_to_module[target]

        description = generate_description(module, args[0], result, target,
                                           args[0].name)

        self.summary.append(description)

        return result