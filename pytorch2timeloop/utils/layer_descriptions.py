"""
A layer description is a representation of a Timeloop workload as a Python dataclass.

The dataclass representation is a lot more convenient than YAML when working in Python: for example, we can easily
define helper properties (like p and q for convolutional layers).

A layer description may use any YAML template (the name of the file that will be used is given by the `problem_template`
attribute). Furthermore, any number of descriptions may use the same template but map the parameters differently.
"""
from functools import reduce
import string
from typing import Optional, Sequence
import pkgutil

import yaml
from dataclasses import dataclass


@dataclass
class LayerDescription:
    name: str

    def get_workload(self):
        f = pkgutil.get_data("pytorch2timeloop",
                             f"utils/{self.problem_template}.yaml")
        return yaml.load(f, Loader=yaml.SafeLoader)

    def to_yaml(self):
        config = self.get_workload()
        config['problem']['shape']['name'] = self.name
        return config


@dataclass
class ConvLayerDescription(LayerDescription):
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

    @property
    def q(self):
        return int((self.w - self.s + 2 * self.w_pad) / self.w_stride) + 1

    @property
    def p(self):
        return int((self.h - self.r + 2 * self.h_pad) / self.h_stride) + 1

    def to_yaml(self):
        dims = list(map(lambda n: self.name + '_' + n, 'GCMRSNPQ'))
        (dim_G, dim_C, dim_M, dim_R, dim_S, dim_N, dim_P, dim_Q) = dims

        in_channels_per_group = self.c // self.g
        out_channels_per_group = self.m // self.g

        config = {
            'shape': {
                'name': self.name,
                'dimensions': dims,
                'coefficients': [
                    {
                        'name': 'Cgroup',
                        'default': in_channels_per_group
                    },
                    {
                        'name': 'Mgroup',
                        'default': out_channels_per_group
                    },
                    {
                        'name': 'Hstride',
                        'default': self.h_stride
                    },
                    {
                        'name': 'Wstride',
                        'default': self.w_stride
                    }
                ],
                'data-spaces': [
                    {
                        'name': self.filter_name,
                        'projection': [
                            [[dim_G]],
                            [[dim_C]],
                            [[dim_M]],
                            [[dim_R]],
                            [[dim_S]]
                        ]
                    },
                    {
                        'name': self.ifmap_name,
                        'projection': [
                            [[dim_N]],
                            [[dim_G, 'Cgroup'], [f'{dim_C}']],
                            [[dim_R], [dim_P, 'Hstride']],
                            [[dim_S], [dim_Q, 'Wstride']]
                        ]
                    },
                    {
                        'name': self.ofmap_name,
                        'projection': [
                            [[dim_N]],
                            [[dim_G, 'Mgroup'], [dim_M]],
                            [[dim_P]],
                            [[dim_Q]]
                        ],
                        'read-write': True
                    }
                ]
            },
            'instance': {
                'G': self.g,
                'C': in_channels_per_group,
                'M': out_channels_per_group,
                'N': self.n,
                'R': self.r,
                'S': self.s,
                'P': self.p,
                'Q': self.q
            }
        }

        return config

    def to_fused_yaml(self):
        dims = list(map(lambda n: self.name + '_' + n, 'GCMRSNPQ'))
        (dim_G, dim_C, dim_M, dim_R, dim_S, dim_N, dim_P, dim_Q) = dims

        in_channels_per_group = self.c // self.g
        out_channels_per_group = self.m // self.g

        config = {
            'shape': {
                'name': self.name,
                'dimensions': dims,
                'data-spaces': [
                    {
                        'name': self.filter_name,
                        'projection': \
                            f'[ {dim_G}, {dim_C}, {dim_M}, {dim_R}, {dim_S} ]'
                    },
                    {
                        'name': self.ifmap_name,
                        'projection': (
                            f'[ {dim_N}, '
                              f'{dim_G}*{in_channels_per_group} + {dim_C}, '
                              f'{dim_R} + {dim_P}*{self.h_stride},  '
                              f'{dim_S} + {dim_Q}*{self.w_stride} ]'
                        )
                    },
                    {
                        'name': self.ofmap_name,
                        'projection': (
                            f'[ {dim_N}, '
                              f'{dim_G}*{out_channels_per_group} + {dim_M}, '
                              f'{dim_P}, '
                              f'{dim_Q} ]'
                        ),
                        'read-write': True
                    }
                ]
            },
            'instance': (
                f'0 <= {dim_G} < {self.g} and '
                f'0 <= {dim_C} < {in_channels_per_group} and '
                f'0 <= {dim_M} < {out_channels_per_group} and '
                f'0 <= {dim_N} < {self.n} and '
                f'0 <= {dim_P} < {self.p} and '
                f'0 <= {dim_Q} < {self.q} and '
                f'0 <= {dim_R} < {self.r} and '
                f'0 <= {dim_S} < {self.s}'
            )
        }

        return config


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
            elif dspace['name'] == 'Outputs':
                dspace['name'] = self.ofmap_name
        return config

    def to_fused_yaml(self):
        dims = list(map(lambda n: self.name + '_' + n, 'CRSNPQ'))
        (dim_C, dim_R, dim_S, dim_N, dim_P, dim_Q) = dims

        config = {
            'shape': {
                'name': self.name,
                'dimensions': dims,
                'data-spaces': [
                    {
                        'name': self.ifmap_name,
                        'projection': (
                            f'[ {dim_N}, '
                              f'{dim_C}, '
                              f'{dim_R} + {dim_P}*{self.h_stride},  '
                              f'{dim_S} + {dim_Q}*{self.w_stride} ]'
                        )
                    },
                    {
                        'name': self.ofmap_name,
                        'projection': (
                            f'[ {dim_N}, '
                              f'{dim_C}, '
                              f'{dim_P}, '
                              f'{dim_Q} ]'
                        ),
                        'read-write': True
                    }
                ]
            },
            'instance': (
                f'0 <= {dim_C} < {self.c} and '
                f'0 <= {dim_N} < {self.n} and '
                f'0 <= {dim_P} < {self.p} and '
                f'0 <= {dim_Q} < {self.q} and '
                f'0 <= {dim_R} < {self.r} and '
                f'0 <= {dim_S} < {self.s}'
            )
        }

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
class BinaryElementwiseFuncDescription(LayerDescription):
    problem_template='binary_elementwise'
    ifmap1_shape: Sequence
    ifmap2_shape: Sequence
    ofmap_shape: Sequence
    ifmap1_name: str
    ifmap2_name: str
    ofmap_name: str

    def to_yaml(self):
        if len(self.ifmap1_shape) < len(self.ofmap_shape):
            n_missing_dims = len(self.ofmap_shape) - len(self.ifmap1_shape)
            self.ifmap1_shape = tuple(
                [1]*n_missing_dims + list(self.ifmap1_shape)
            )
        if len(self.ifmap2_shape) < len(self.ofmap_shape):
            n_missing_dims = len(self.ofmap_shape) - len(self.ifmap2_shape)
            self.ifmap2_shape = tuple(
                [1]*n_missing_dims + list(self.ifmap2_shape)
            )
        assert(len(self.ifmap1_shape) == len(self.ofmap_shape))
        assert(len(self.ifmap2_shape) == len(self.ofmap_shape))

        config = super().to_yaml()

        dims = list(string.ascii_uppercase[:len(self.ofmap_shape)])

        for dspace in config['problem']['shape']['data-spaces']:
            if dspace['name'] == 'Input1':
                dspace['name'] = self.ifmap1_name
                dspace['projection'] = []
                for d, size in zip(dims, self.ifmap1_shape):
                    if size > 1:
                        dspace['projection'].append([[d]])
            elif dspace['name'] == 'Input2':
                dspace['name'] = self.ifmap2_name
                dspace['projection'] = []
                for d, size in zip(dims, self.ifmap2_shape):
                    if size > 1:
                        dspace['projection'].append([[d]])
            elif dspace['name'] == 'Outputs':
                dspace['name'] = self.ofmap_name
                dspace['projection'] = list(map(
                    lambda d: [[d]],
                    dims
                ))

        config['problem']['shape']['dimensions'] = dims

        config['problem']['instance'] = {}
        for dim, size in zip(dims, self.ifmap1_shape):
            config['problem']['instance'][dim] = size

        return config

    def to_fused_yaml(self):
        if len(self.ifmap1_shape) < len(self.ofmap_shape):
            n_missing_dims = len(self.ofmap_shape) - len(self.ifmap1_shape)
            self.ifmap1_shape = tuple(
                [1]*n_missing_dims + list(self.ifmap1_shape)
            )
        if len(self.ifmap2_shape) < len(self.ofmap_shape):
            n_missing_dims = len(self.ofmap_shape) - len(self.ifmap2_shape)
            self.ifmap2_shape = tuple(
                [1]*n_missing_dims + list(self.ifmap2_shape)
            )
        assert(len(self.ifmap1_shape) == len(self.ofmap_shape))
        assert(len(self.ifmap2_shape) == len(self.ofmap_shape))

        dims = list(string.ascii_uppercase[:len(self.ofmap_shape)])
        bounds = []
        for dim_name, dim_size in zip(dims, self.ifmap1_shape):
            bounds.append(f'0 <= {dim_name} < {dim_size}')

        config = {
            'shape': {
                'name': self.name,
                'dimensions': dims,
                'data-spaces': [
                    {
                        'name': self.ifmap1_name,
                        'projection': '[ ' + ', '.join(dims) + ' ]'
                    },
                    {
                        'name': self.ifmap2_name,
                        'projection': '[ ' + ', '.join(dims) + ' ]'
                    },
                    {
                        'name': self.ofmap_name,
                        'projection': '[ ' + ', '.join(dims) + ' ]',
                        'read-write': True
                    }
                ]
            },
            'instance': ' and '.join(bounds)
        }

        return config


@dataclass
class MatmulFuncDescription(LayerDescription):
    problem_template = "matmul"
    m: int
    n: int
    k: int
    ifmap1_name: str
    ifmap2_name: str
    ofmap_name: str
    extra_dims: Optional[tuple] = None

    def to_yaml(self):
        config = super().to_yaml()

        if self.extra_dims is not None:
            dims = tuple(string.ascii_uppercase[:len(self.extra_dims)])
        else:
            dims = tuple()
            self.extra_dims = tuple()

        for dspace in config['problem']['shape']['data-spaces']:
            if dspace['name'] == 'Input1':
                dspace['name'] = self.ifmap1_name
            elif dspace['name'] == 'Input2':
                dspace['name'] = self.ifmap2_name
            elif dspace['name'] == 'Outputs':
                dspace['name'] = self.ofmap_name
            proj_dims = list(map(lambda d: [[d]], dims))
            dspace['projection'] = proj_dims + dspace['projection']

        config['problem']['instance']['K'] = self.k
        config['problem']['instance']['M'] = self.m
        config['problem']['instance']['N'] = self.n
        config['problem']['shape']['name'] = self.name

        for dim, size in zip(dims, self.extra_dims):
            config['problem']['instance'][dim] = size

        return config


@dataclass
class SoftmaxFuncDescription(LayerDescription):
    problem_template = 'softmax'
    ifmap_shape: tuple
    ofmap_shape: tuple
    ifmap_name: str
    ofmap_name: str
    softmax_dim: int

    def to_yaml(self):
        config = super().to_yaml()

        dims = tuple(string.ascii_uppercase[:len(self.ifmap_shape)+1])

        for dspace in config['problem']['shape']['data-spaces']:
            if dspace['name'] == 'Input':
                dspace['name'] = self.ifmap_name
                dspace['projection'] = list(map(
                    lambda d: [[d]],
                    dims[:-1]
                ))
            elif dspace['name'] == 'Output':
                dspace['name'] = self.ofmap_name
                dspace['projection'] = list(map(
                    lambda d: [[d]],
                    dims[:-1]
                ))
                dspace['projection'][self.softmax_dim] = [[dims[-1]]]

        instance = {}
        for dim, size in zip(dims[:-1], self.ifmap_shape):
            instance[dim] = size
        instance[dims[-1]] = self.ofmap_shape[self.softmax_dim]
        config['problem']['instance'] = instance

        return config


@dataclass
class ViewFuncDescription(LayerDescription):
    problem_template = 'view'
    ifmap_shape: tuple
    ofmap_shape: tuple
    ifmap_name: str
    ofmap_name: str

    def to_yaml(self):
        raise NotImplementedError('cannot be implemented in old Timeloop spec')

    def to_fused_yaml(self):
        product = lambda l: reduce(lambda x, y: x*y, l)
        assert(product(self.ifmap_shape) == product(self.ofmap_shape))

        n_ofmap_dims = len(self.ofmap_shape)
        ofmap_dims = list(string.ascii_uppercase[:n_ofmap_dims])

        bounds = []
        for dim_name, dim_size in zip(ofmap_dims, self.ofmap_shape):
            bounds.append(f'0 <= {dim_name} < {dim_size}')
        
        terms = []
        cur_size = 1
        for dim, dim_size in reversed(list(zip(ofmap_dims, self.ofmap_shape))):
            terms.append(f'{dim}*{cur_size}')
            cur_size *= dim_size
        linearized_ofmaps = ' + '.join(terms)

        ifmap_terms = []
        cur_size = 1
        for dim_size in reversed(self.ifmap_shape):
            ifmap_terms.append(
                f'floor({linearized_ofmaps}/{cur_size})%{dim_size}'
            )
            cur_size *= dim_size
        ifmap_terms.reverse()

        config = {
            'shape': {
                'name': self.name,
                'dimensions': ofmap_dims,
                'data-spaces': [
                    {
                        'name': self.ifmap_name,
                        'projection': '[ ' + ', '.join(ifmap_terms) + ' ]'
                    },
                    {
                        'name': self.ofmap_name,
                        'projection': '[ ' + ', '.join(ofmap_dims) + ' ]',
                        'read-write': True
                    }
                ]
            },
            'instance': ' and '.join(bounds)
        }

        return config
