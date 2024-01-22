""" Convert Trained PyTorch Models to Workloads """

import logging
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.fx as fx

import yaml

from pytorch2timeloop.utils.interpreter import Converter

logger = logging.getLogger(__name__)


def convert_model_with_sample_input(model: nn.Module,
                                    sample_input: Any,
                                    batch_size: int,
                                    model_name: str,
                                    save_dir: Path,
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

    layer_data = _make_summary(model, sample_input)
    _convert_from_layer_data(layer_data, model_name, save_dir)


def convert_model(model: nn.Module, input_size: tuple, batch_size: int,
                  model_name: str, save_dir: Path,
                  fuse=False, convert_fc=False,
                  ignored_func=None,
                  exception_module_names=()):
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
    layer_data = _make_summary(model, sample_input, ignored_func=ignored_func)
    _convert_from_layer_data(layer_data, model_name, save_dir, exception_module_names, fuse=fuse)


def _convert_from_layer_data(layer_data, model_name, save_dir, exception_module_names=(), fuse=False):
    outdir = os.path.join(save_dir, model_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    layer_data = [
        p for p in layer_data if not any(
            e.lower() in p.name.lower() or 
            e.lower() in p.__class__.__name__.lower() 
            for e in exception_module_names)]
    if fuse:
        problems = []
        for i in range(0, len(layer_data)):
            problem = layer_data[i]
            problems.append(problem.to_fused_yaml())
        file_name = model_name + '.yaml'
        file_path = os.path.abspath(os.path.join(save_dir, model_name, file_name))
        with open(file_path, 'w') as f:
            f.write(yaml.dump(
                {
                    'problem': problems
                }
            ))
    else:
        # make the problem file for each layer
        for i in range(0, len(layer_data)):
            problem = layer_data[i]
            file_name = '[layer' + str(i+1) + ']' + problem.name + '.yaml'
            file_path = os.path.abspath(os.path.join(save_dir, model_name, file_name))
            with open(file_path, 'w') as f:
                f.write(yaml.dump(problem.to_yaml()))

    logger.info("conversion complete!\n")

def _make_summary(model, sample_input, ignored_func):
    converter = Converter(fx.symbolic_trace(model), ignored_func=ignored_func)
    converter.run(sample_input)
    return converter.summary
