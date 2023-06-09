""" Convert Trained PyTorch Models to Workloads """

import logging
import os
from typing import Any

import torch
import yaml

from pytorch2timeloop.utils.hooks import hook_for
from torch import nn

logger = logging.getLogger(__name__)


def extract_layer_data(model: nn.Module,
                       input_size: tuple, batch_size: int,
                       exception_module_names=[]):
    """
    Convert a general PyTorch model to Timeloop problem representations.

    The difference with `convert_model` is that this does not create
    files.

    :param model: the PyTorch CNN model
    :param input_size: a tuple representing the input size
    :param batch_size: the batch size
    :param convert_fc: whether to convert fully connected layers
    :param exception_module_names: a list of fragments of module names
        to ignore (can be a prefix, suffix, or infix).
    """

    sample_input = torch.rand(2, *input_size).type(torch.FloatTensor)
    return _new_extract_layer_data(
        model,
        sample_input,
        batch_size=batch_size,
        exception_module_names=exception_module_names
    )


def convert_model_with_sample_input_v2(model: nn.Module,
                                       sample_input: Any, batch_size: int,
                                       model_name: str,
                                       save_dir: str,
                                       exception_module_names=[]):
    """
    Convert a general PyTorch model to Timeloop problem files.

    Currently, only common CNNs and the BERT transformer from
    `transformers` are supported, but it is easy to add support for new
    DNNs. See documentation in `utils/hooks.py` for more on supporting
    new PyTorch module types. This interface is more general than
    `convert_model()` and should be preferred for new code.

    :param model: the PyTorch CNN model
    :param sample_input:
    :param batch_size: the batch size
    :param model_name: the name of the model, which will become the name
        of the subdirectory of `save_dir` with the problem files
    :param save_dir: the directory to save the output in
    :param exception_module_names: a list of fragments of module names
        to ignore (can be a prefix, suffix, or infix).
    """
    logger.info("converting {} in {} model ...".format("all", model_name))

    layer_data = _new_extract_layer_data(
        model,
        sample_input,
        batch_size=batch_size,
        exception_module_names=exception_module_names
    )
    _convert_from_layer_data(layer_data, model_name, save_dir)


def convert_model_with_sample_input(model: nn.Module,
                                    sample_input: Any,
                                    batch_size: int,
                                    model_name: str,
                                    save_dir: str,
                                    exception_module_names=[]):
    """
    Convert a general PyTorch model to Timeloop problem files.

    Currently, only common CNNs and the BERT transformer from
    `transformers` are supported, but it is easy to add support for new
    DNNs. See documentation in `utils/hooks.py` for more on supporting
    new PyTorch module types. This interface is more general than
    `convert_model()` and should be preferred for new code.

    :param model: the PyTorch CNN model
    :param sample_input:
    :param batch_size: the batch size
    :param model_name: the name of the model, which will become the name
        of the subdirectory of `save_dir` with the problem files
    :param save_dir: the directory to save the output in
    :param exception_module_names: a list of fragments of module names
        to ignore (can be a prefix, suffix, or infix).
    """
    logger.info("converting {} in {} model ...".format("all", model_name))

    layer_data = _extract_layer_data(model, sample_input, convert_fc=True,
                                     batch_size=batch_size,
                                     exception_module_names=exception_module_names)
    _convert_from_layer_data(layer_data, model_name, save_dir)


def convert_model(model: nn.Module, input_size: tuple, batch_size: int,
                  model_name: str, save_dir: str,
                  convert_fc=False, exception_module_names=[]):
    """
    Convert a PyTorch CNN model to Timeloop problem files.

    This is the original interface to this library from 0.1.
    The primary difference between it and `convert_model_with_sample_input`
    is that it accepts an extra parameter (`convert_fc`) and accepts an
    input size parameter as a tuple, rather than a sample input.

    :param model: the PyTorch CNN model
    :param input_size: a tuple representing the input size
    :param batch_size: the batch size
    :param model_name: the name of the model, which will become the name
        of the subdirectory of `save_dir` with the problem files
    :param save_dir: the directory to save the output in
    :param convert_fc: whether to convert fully connected layers
    :param exception_module_names: a list of fragments of module names
        to ignore (can be a prefix, suffix, or infix).
    """
    logger.info(
        "converting {} in {} model ...".format(
            "nn.Conv2d" if not convert_fc else "nn.Conv2d and nn.Linear",
            model_name
        )
    )
    sample_input = torch.rand(2, *input_size).type(torch.FloatTensor)
    layer_data = _extract_layer_data(model, sample_input, convert_fc=convert_fc,
                                     exception_module_names=exception_module_names,
                                     batch_size=batch_size)
    _convert_from_layer_data(layer_data, model_name, save_dir)


def _make_summary(model, sample_input, convert_fc=False, batch_size=1):
    model.eval()
    # create properties
    summary = []
    hooks = []

    def register_hook(module):
        hook = hook_for(module, summary, batch_size, convert_fc=convert_fc)
        if hook is not None:
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    try:
        model(**sample_input)
    except TypeError:
        try:
            model(*sample_input)
        except TypeError:
            model(sample_input)

    # remove these hooks
    for h in hooks:
        h.remove()

    return summary


def _extract_layer_data(model, sample_inputs,
                        convert_fc=False, exception_module_names=[],
                        batch_size=1):
    conv_list = []
    for name, layer in model.named_modules():
        for exception in exception_module_names:
            if name not in exception:
                conv_list.append(layer)

    summary = _make_summary(model, sample_inputs, convert_fc=convert_fc,
                            batch_size=batch_size)
    return summary


def _convert_from_layer_data(layer_data, model_name, save_dir):
    outdir = os.path.join(save_dir, model_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # make the problem file for each layer
    for i in range(0, len(layer_data)):
        problem = layer_data[i]
        file_name = model_name + '_' + 'layer' + str(i+1) + '.yaml'
        file_path = os.path.abspath(os.path.join(save_dir, model_name, file_name))
        with open(file_path, 'w') as f:
            f.write(yaml.dump(problem.to_yaml()))

    logger.info("conversion complete!\n")

def _new_make_summary(model, sample_input, batch_size):
    # TODO: rewrite with FX interpreter
    raise NotImplementedError()

def _new_extract_layer_data(model, sample_inputs,
                            exception_module_names=[], batch_size=1):
    summary = _new_make_summary(model, sample_inputs, batch_size=batch_size)
    return summary