"""
A layer description is a representation of a Timeloop workload as a Python dataclass.

The dataclass representation is a lot more convenient than YAML when working in Python: for example, we can easily
define helper properties (like p and q for convolutional layers).

A layer description may use any YAML template (the name of the file that will be used is given by the `problem_template`
attribute). Furthermore, any number of descriptions may use the same template but map the parameters differently.
"""
import string
from typing import Optional, Sequence
import pkgutil

import yaml
from dataclasses import dataclass


@dataclass
class LayerDescription:
    name: Optional[str]

    def get_workload(self):
        f = pkgutil.get_data("pytorch2timeloop",
                             f"utils/{self.problem_template}.yaml")
        return yaml.load(f, Loader=yaml.SafeLoader)

    def to_yaml(self):
        config = self.get_workload()
        config['problem']['shape']['name'] = self.name
        return config


class BaseConvLayerDescription(LayerDescription):
    @property
    def q(self):
        return int((self.w - self.s + 2 * self.w_pad) / self.w_stride) + 1

    @property
    def p(self):
        return int((self.h - self.r + 2 * self.h_pad) / self.h_stride) + 1

    def to_yaml(self):
        config = super().to_yaml()
        assert self.n >= 0
        assert self.p >= 0
        config['problem']['instance']['R'] = self.r
        config['problem']['instance']['S'] = self.s
        config['problem']['instance']['P'] = self.p
        config['problem']['instance']['Q'] = self.q
        config['problem']['instance']['C'] = self.c
        config['problem']['instance']['N'] = self.n
        config['problem']['instance']['Wstride'] = self.w_stride
        config['problem']['instance']['Hstride'] = self.h_stride

        for dspace in config['problem']['shape']['data-spaces']:
            if dspace['name'] == 'Inputs':
                dspace['name'] = self.ifmap_name
            if dspace['name'] == 'Weights':
                dspace['name'] = self.filter_name
            if dspace['name'] == 'Outputs':
                dspace['name'] = self.ofmap_name
        return config


@dataclass
class ConvLayerDescription(BaseConvLayerDescription):
    problem_template = "convolution"
    m: int

    w: int
    h: int
    c: int
    n: int
    s: int
    r: int
    w_pad: int
    h_pad: int
    w_stride: int
    h_stride: int
    ifmap_name: str
    filter_name: str
    ofmap_name: str

    def to_yaml(self):
        config = super().to_yaml()
        config['problem']['instance']['M'] = self.m
        return config

@dataclass
class GroupedConvLayerDescription(BaseConvLayerDescription):
    problem_template = "grouped_convolution"
    g: int
    m: int

    w: int
    h: int
    c: int
    n: int
    s: int
    r: int
    w_pad: int
    h_pad: int
    w_stride: int
    h_stride: int
    ifmap_name: str
    filter_name: str
    ofmap_name: str

    def to_yaml(self):
        config = super().to_yaml()
        config['problem']['instance']['G'] = self.g
        config['problem']['instance']['M'] = self.m
        return config

@dataclass()
class DepthWiseConvLayerDescription(BaseConvLayerDescription):
    problem_template = "depth_wise_convolution"
    w: int
    h: int
    c: int
    n: int
    s: int
    r: int
    w_pad: int
    h_pad: int
    w_stride: int
    h_stride: int
    ifmap_name: str
    filter_name: str
    ofmap_name: str


@dataclass
class MaxPoolLayerDescription(LayerDescription):
    @property
    def q(self):
        return int((self.w - self.s + 2 * self.w_pad) / self.w_stride) + 1

    @property
    def p(self):
        return int((self.h - self.r + 2 * self.h_pad) / self.h_stride) + 1

    problem_template = 'pool'

    w: int
    h: int
    c: int
    n: int
    s: int
    r: int
    w_pad: int
    h_pad: int
    w_stride: int
    h_stride: int
    ifmap_name: str
    ofmap_name: str

    def to_yaml(self):
        config = super().to_yaml()
        config['problem']['instance']['R'] = self.r
        config['problem']['instance']['S'] = self.s
        config['problem']['instance']['P'] = self.p
        config['problem']['instance']['Q'] = self.q
        config['problem']['instance']['C'] = self.c
        config['problem']['instance']['N'] = self.n
        config['problem']['instance']['Wstride'] = self.w_stride
        config['problem']['instance']['Hstride'] = self.h_stride

        for dspace in config['problem']['shape']['data-spaces']:
            if dspace['name'] == 'Inputs':
                dspace['name'] = self.ifmap_name
            if dspace['name'] == 'Outputs':
                dspace['name'] = self.ofmap_name
        return config


@dataclass
class MatrixMatrixMultiplyLayerDescription(LayerDescription):
    name: Optional[str]
    problem_template = "convolution"
    m: int
    n: int
    k: int
    batch_size: int

    def to_yaml(self):
        config = super().to_yaml()
        config['problem']['instance']['R'] = 1
        config['problem']['instance']['S'] = self.k
        config['problem']['instance']['P'] = self.m
        config['problem']['instance']['Q'] = 1
        config['problem']['instance']['C'] = 1
        config['problem']['instance']['M'] = self.n
        config['problem']['instance']['N'] = self.batch_size
        config['problem']['instance']['Wstride'] = 1
        config['problem']['instance']['Hstride'] = 1
        config['problem']['shape']['name'] = self.name
        return config


@dataclass
class AddFuncDescription(LayerDescription):
    problem_template='add'
    ifmap_shape: Sequence
    ofmap_shape: Sequence
    ifmap_name: str
    ofmap_name: str

    def to_yaml(self):
        assert(self.ifmap_shape == self.ofmap_shape)
        config = super().to_yaml()

        dims = list(string.ascii_uppercase[:len(self.ifmap_shape)])

        for dspace in config['problem']['shape']['data-spaces']:
            if dspace['name'] == 'Inputs':
                dspace['name'] = self.ifmap_name
            if dspace['name'] == 'Outputs':
                dspace['name'] = self.ofmap_name
            dspace['projection'] = list(map(
                lambda d: [[d]],
                dims
            ))
        
        config['problem']['shape']['dimensions'] = dims

        config['problem']['instance'] = {}
        for dim, size in zip(dims, self.ifmap_shape):
            config['problem']['instance'][dim] = size

        return config
