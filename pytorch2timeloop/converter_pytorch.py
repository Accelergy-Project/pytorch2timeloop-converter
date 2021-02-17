
""" Convert Trained PyTorch Models to Workloads """

import torchvision
import re
import os, inspect, sys
import functools
import yaml

import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

from pytorch2timeloop.utils.construct_workloads import *

""" 
This section of code is taken directly from the github repo https://github.com/sksq96/pytorch-summary
and modified to fit our specific needs.  The code performs a forward pass through the model with hooks
in order to track the output shapes of different layers throughout a model.  This is a useful alternative
to hand calculation and helps avoid bugs, especially for large, complex networks like ResNet.
"""
def make_summary(model, input_size, convert_fc=False, batch_size=-1, \
                 device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu"), dtypes=None):
    # Modified (Dec 17, 2020) - Kyungmi
    # Set the model to evaluation mode
    # (inference graph can differ from training graph when there are some modules not used during inference)
    # (e.g., auxiliary classifier in Inception v3)
    model.eval()
    
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        # Modified (Dec 17, 2020) - Kyungmi
        # only append hooks to Conv2d layers in the model
        if (
            # not isinstance(module, nn.Sequential)
            # and not isinstance(module, nn.ModuleList)
            isinstance(module, nn.Conv2d) or (isinstance(module, nn.Linear) and convert_fc)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return summary



''' 
Designed to extract info about convolutional layers from a model.
Returns a nested list with information about each convolutional layer
in the form of [in_ch, out_ch, kernel_w, kernel_h, w_stride, h_stride, w_pad, h_pad]
'''
def convert_model(model, input_size, batch_size, model_name, save_dir, convert_fc=False, exception_module_names=[]):

    print("converting {} in {} model ...".format("nn.Conv2d" if not convert_fc else "nn.Conv2d and nn.Linear", model_name))

    layer_data  = extract_layer_data(model, input_size, convert_fc, exception_module_names)
    layer_list = []

    for layer in layer_data:
        if layer_data[layer]['mode'] == 'linear':
            W, H = 1, 1
            C, M = layer_data[layer]['in_channels'], layer_data[layer]['out_channels']
            S, R = layer_data[layer]['kernel_width'], layer_data[layer]['kernel_height']
            Wpad, Hpad = layer_data[layer]['padding_width'], layer_data[layer]['padding_height']
            Wstride, Hstride = layer_data[layer]['stride_width'], layer_data[layer]['stride_height']
            # G, B = layer_data[layer]['groups'], layer_data[layer]['bias']
            G = layer_data[layer]['groups']
            N = batch_size
            Mode = layer_data[layer]['mode']
        else:
            W, H = layer_data[layer]['input_shape'][2:]
            C, M = layer_data[layer]['in_channels'], layer_data[layer]['out_channels']
            S, R = layer_data[layer]['kernel_width'], layer_data[layer]['kernel_height']
            Wpad, Hpad = layer_data[layer]['padding_width'], layer_data[layer]['padding_height']
            Wstride, Hstride = layer_data[layer]['stride_width'], layer_data[layer]['stride_height']
            # G, B = layer_data[layer]['groups'], layer_data[layer]['bias']
            G = layer_data[layer]['groups']
            N = batch_size
            Mode = layer_data[layer]['mode']
        layer_entry = [Mode, W, H, C, N, M, S, R, Wpad, Hpad, Wstride, Hstride, G]#, B]
        layer_list.append(layer_entry)
    
    outdir = os.path.join(save_dir, model_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Modify (Feb 1, 2021) - Kyungmi
    # Remove relative path calling and replace with util function
    # this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    # this_directory = os.path.dirname(this_file_path)
    # config_abspath_conv =  os.path.join(this_directory, 'utils/convolution.yaml')
    # config_abspath_depth = os.path.join(this_directory, 'utils/depth_wise_convolution.yaml')

    # test for valid config files
    # with open(config_abspath_conv, 'r') as f:
    #     config_conv = yaml.load(f, Loader = yaml.SafeLoader)
    # with open(config_abspath_depth, 'r') as f:
    #     config_depth = yaml.load(f, Loader = yaml.SafeLoader)
    
    # make the problem file for each layer
    for i in range(0, len(layer_list)):
        problem = layer_list[i]
        layer_type = problem[0]
        file_name = model_name + '_' + 'layer' + str(i+1) + '.yaml'
        file_path = os.path.abspath(os.path.join(save_dir, model_name, file_name))
        if layer_type == 'norm-conv' or layer_type == 'linear':
            rewrite_workload_bounds(file_path, problem)
        elif layer_type == 'depth-wise':
            rewrite_workload_bounds(file_path, problem)
        else:
            print("Error: DNN Layer Type {} Not Supported".format(layer_type))
            return
        
    print("conversion complete!\n")

    
def extract_layer_data(model, input_size, convert_fc=False, exception_module_names=[]):
    data = {}
    layer_number = 1
    
    """
    Modified (Dec 17, 2020) - Kyungmi
    Directly obtain a list of nn.Conv2d modules in the model. 
    If there is a nn.Conv2d module that should not be counted during the inference
    (e.g., Auxiliary classification layer in Inceptionv3), 
    include the identifier for a such layer (e.g., 'Aux') in exception_module_names (a list of str). 
    """
    conv_list = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or (convert_fc and isinstance(layer, nn.Linear)):
            if exception_module_names is not None:
                flag = False
                for exception in exception_module_names:
                    if exception in name:
                        flag = True
                        break
                if not flag:
                    conv_list.append(layer)
            else:
                conv_list.append(layer)
            
    for conv in conv_list:

        data[layer_number] = {
                'mode':             "norm-conv",
                'input_shape':      [0, 0, 0, 0],
                'output_shape':     [0, 0, 0, 0],
                'in_channels':      0,
                'out_channels':     0,
                'kernel_width':     0,
                'kernel_height':    0,
                'stride_width':     1,
                'stride_height':    1,
                'padding_width':    0,
                'padding_height':   0,
                'groups':           1,
                # 'bias':             True,
        }
        
        if isinstance(conv, nn.Conv2d):
        
            data[layer_number]['in_channels'] = conv.in_channels
            data[layer_number]['out_channels'] = conv.out_channels
            data[layer_number]['kernel_width'] = conv.kernel_size[0]
            data[layer_number]['kernel_height'] = conv.kernel_size[1]
            data[layer_number]['stride_width'] = conv.stride[0]
            data[layer_number]['stride_height'] = conv.stride[1]
            data[layer_number]['padding_width'] =  conv.padding[0]
            data[layer_number]['padding_height'] = conv.padding[1]
            data[layer_number]['groups'] = conv.groups

            data[layer_number]['mode'] = 'norm-conv'
            if data[layer_number]['groups'] > 1 and data[layer_number]['groups'] == data[layer_number]['in_channels']:
                data[layer_number]['mode'] = 'depth-wise'
                data[layer_number]['in_channels'] = 1
                data[layer_number]['out_channels'] = 1
                
        elif isinstance(conv, nn.Linear):
            
            # Convert Linear to Conv (https://cs231n.github.io/convolutional-networks/#fc)
            # in_channels = conv.in_features
            # out_channels = conv.out_features
            # kernel = 1x1, stride = 1x1, padding = 0x0, groups = 1
            # mode = 'linear'
            data[layer_number]['in_channels'] = conv.in_features
            data[layer_number]['out_channels'] = conv.out_features
            data[layer_number]['kernel_width'] = 1
            data[layer_number]['kernel_height'] = 1
            data[layer_number]['stride_width'] = 1
            data[layer_number]['stride_height'] = 1
            data[layer_number]['padding_width'] = 0
            data[layer_number]['padding_height'] = 0
            data[layer_number]['groups'] = 1
            data[layer_number]['mode'] = 'linear'

        layer_number += 1
        
    layer_number = 1
    summary = make_summary(model, input_size, convert_fc)

    assert len(data.keys()) == len([layer for layer in summary if ("Conv2d" in layer or ("Linear" in layer and convert_fc))]), \
            "Different number of conv layers detected by filter and io"
    
    for layer in summary:
        if "Conv2d" in layer or ("Linear" in layer and convert_fc):
            data[layer_number]["input_shape"] = summary[layer]["input_shape"]
            data[layer_number]["output_shape"] = summary[layer]["output_shape"]
            layer_number += 1
   
    return data





