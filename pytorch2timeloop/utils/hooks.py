"""
Definitions of forward hooks for various PyTorch layer types, to extract `LayerDescription`s from them during evaluation.

For many layer types, such as 2D convolutions and self-attention mechanisms, the layer itself does not know all of the
information needed to generate a Timeloop workload: for example, a convolutional layer does not explicitly define its
input size. As a result, we need to extract this information while _evaluating_ the model.

The easiest mechanism for doing this is a PyTorch forward hook. This file defines hooks for various layer types,
with a primary interface consisting of the function `hook_for()`, which returns a hook for the given layer.

To add support for a new layer type, add a new hook type and return it from hook_for() with the appropriate conditions.
You may also need to add a new `LayerDescription` if the layer is very different from the ones that are already here.
"""

from typing import Optional, Callable, Any

import torch.nn as nn
import transformers.models.distilbert.modeling_distilbert

from pytorch2timeloop.utils.layer_descriptions import DepthWiseConvLayerDescription, ConvLayerDescription, MatrixMatrixMultiplyLayerDescription


def _null_hook(summary, batch_size):
    """
    An empty hook, for layers that we want to ignore without error (like ReLU)
    """
    def hook(module, input, output):
        return
    return hook


def _conv_hook(summary, batch_size):
    """
    A hook for convolutional (including depth-wise convolutional) layers, based on nn.Conv2d.

    :param summary: the summary list we are adding to
    :param batch_size: the input batch size
    :return: a PyTorch module forward hook to collect a `LayerDescription` about this convolutional layer
    """
    def hook(module, input, output):
        input_shape = input[0].size()
        if module.groups > 1 and module.groups == module.in_channels:
            # Depth-wise convolution
            description = DepthWiseConvLayerDescription(
                w=input_shape[2],
                h=input_shape[3],
                c=module.in_channels,
                s=module.kernel_size[0],
                r=module.kernel_size[1],
                w_stride=module.stride[0],
                h_stride=module.stride[1],
                w_pad=module.padding[0],
                h_pad=module.padding[1],
                n=batch_size,
                name="conv_layer"
            )
        else:
            description = ConvLayerDescription(
                w=input_shape[2],
                h=input_shape[3],
                c=module.in_channels,
                m=module.out_channels,
                s=module.kernel_size[0],
                r=module.kernel_size[1],
                w_stride=module.stride[0],
                h_stride=module.stride[1],
                w_pad=module.padding[0],
                h_pad=module.padding[1],
                n=batch_size,
                name="conv_layer"
            )
        summary.append(description)

    return hook


def _linear_hook(summary, batch_size):
    """
    A hook for linear (i.e., fully connected) layers, based on nn.Linear.

    :param summary: the summary list we are adding to
    :param batch_size: the input batch size
    :return: a PyTorch module forward hook to collect a `LayerDescription` about this fully connected layer
    """
    def hook(module, input, output):
        print(str(id(module)) + str(module))
        input_size = input[0].size()
        assert input_size[1] >= 0
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
            n=batch_size,
            name="linear"
        )
        summary.append(description)

    return hook


def _layer_norm_hook(summary, batch_size):
    """
    A hook for layer norm layers, based on nn.LayerNorm.

    :param summary: the summary list we are adding to
    :param batch_size: the input batch size
    :return: a PyTorch module forward hook to collect a `LayerDescription` about this layer norm layer.
    """

    def hook(module, input, output):
        if module.elementwise_affine:
            print(str(id(module)) + str(module))
            input_shape = input[0].size()
            assert input_shape[1] >= 0
            description = ConvLayerDescription(
                h=input_shape[2],
                w=1,
                c=1,
                m=1,
                r=input_shape[2],
                s=1,
                #s=input_shape[1],
                #r=input_shape[2],
                w_stride=1,
                h_stride=1,
                w_pad=0,
                h_pad=0,
                n=batch_size * input_shape[1],
                name="layer_norm"
            )
        summary.append(description)

    return hook

def _albert_multihead_self_attention(summary, batch_size):
    """
    A hook for multi-head self-attention layers.

    Currently, this is designed only to extract data from the self-attention layer defined in
    `transformers.models.bert.modeling_bert.BertSelfAttention`. It should be quite simple to adapt to other
    transformers with similar self-attention mechanisms, though.

    :param summary: the summary list we are adding to
    :param batch_size: the input batch size
    :return: a PyTorch module forward hook to collect a `LayerDescription` about this multi-head self-attention layer
    """
    def hook(module, input, output):
        assert input != ()
        x = input[0]
        head_size = module.attention_head_size
        sequence_length = x.shape[1]
        scores = MatrixMatrixMultiplyLayerDescription(
            m=sequence_length,
            k=head_size,
            n=sequence_length,
            batch_size=batch_size * module.num_attention_heads,
            name="attention_scores"
        )
        context = MatrixMatrixMultiplyLayerDescription(
            m=sequence_length,
            k=sequence_length,
            n=head_size,
            batch_size=batch_size * module.num_attention_heads,
            name="attention_context"
        )

        input_shape = input[0].shape
        print(str(id(module.dense)) + str(module.dense))
        #print(str(id(module.LayerNorm)) + str(module.LayerNorm))
        ffn_desc = ConvLayerDescription(
            w=1,
            h=1,
            c=module.dense.in_features,
            m=module.dense.out_features,
            s=1,
            r=1,
            w_stride=1,
            h_stride=1,
            w_pad=0,
            h_pad=0,
            n=batch_size * input_shape[1],
            name="linear"
        )

        """
        layer_norm = ConvLayerDescription(
            h=input_shape[2],
            w=1,
            c=1,
            m=1,
            r=input_shape[2],
            s=1,
            w_stride=1,
            h_stride=1,
            w_pad=0,
            h_pad=0,
            n=batch_size * input_shape[1],
            name="layer_norm"
        )
        """

        summary.append(scores)
        summary.append(context)
        summary.append(ffn_desc)
        #summary.append(layer_norm)

    return hook

def _albert_layer(summary, batch_size):
    def hook(module, input, output):
        print(str(id(module.ffn)) + str(module.ffn))
        #print(str(id(module.full_layer_layer_norm)) + str(module.full_layer_layer_norm))
        #  module.full_layer_layer_norm
        #import pdb; pdb.set_trace()
        """
        input_shape = input[0].size()
        assert input_shape[1] >= 0
        description = ConvLayerDescription(
            h=input_shape[2],
            w=1,
            c=1,
            m=1,
            r=input_shape[2],
            s=1,
            w_stride=1,
            h_stride=1,
            w_pad=0,
            h_pad=0,
            n=batch_size * input_shape[1],
            name="layer_norm"
        )

        summary.append(description)

        input_size = input[0].size()
        assert input_size[1] >= 0
        description = ConvLayerDescription(
            w=1,
            h=1,
            c=module.ffn.in_features,
            m=module.ffn.out_features,
            s=1,
            r=1,
            w_stride=1,
            h_stride=1,
            w_pad=0,
            h_pad=0,
            n=batch_size * input_size[1],
            name="linear"
        )

        summary.append(description)
        """

    return hook 

def _multihead_self_attention(summary, batch_size):
    """
    A hook for multi-head self-attention layers.

    Currently, this is designed only to extract data from the self-attention layer defined in
    `transformers.models.bert.modeling_bert.BertSelfAttention`. It should be quite simple to adapt to other
    transformers with similar self-attention mechanisms, though.

    :param summary: the summary list we are adding to
    :param batch_size: the input batch size
    :return: a PyTorch module forward hook to collect a `LayerDescription` about this multi-head self-attention layer
    """
    def hook(module, input, output):
        assert input != ()
        x = input[0]
        print(id(module))
        head_size = module.attention_head_size
        sequence_length = x.shape[1]
        scores = MatrixMatrixMultiplyLayerDescription(
            m=sequence_length,
            k=head_size,
            n=sequence_length,
            batch_size=batch_size * module.num_attention_heads,
            name="attention_scores"
        )
        context = MatrixMatrixMultiplyLayerDescription(
            m=sequence_length,
            k=sequence_length,
            n=head_size,
            batch_size=batch_size * module.num_attention_heads,
            name="attention_context"
        )

        summary.append(scores)
        summary.append(context)

    return hook


"""
Layer types that should be considered "null ops" (i.e., that should not produce a layer file).
This can be safely extended to reduce the amount of noise printed to the terminal when generating layer files.
"""
null_ops = (
    nn.Dropout,
    nn.Embedding,
    nn.ReLU,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
    nn.Sequential,
    nn.ModuleList,
    nn.Tanh,
    transformers.models.bert.modeling_bert.BertSelfOutput,
    transformers.models.bert.modeling_bert.BertEmbeddings,
    #transformers.models.bert.modeling_bert.BertIntermediate,
    transformers.models.bert.modeling_bert.BertOutput,
    transformers.models.bert.modeling_bert.BertAttention,
    transformers.models.bert.modeling_bert.BertLayer,
    transformers.models.bert.modeling_bert.BertEncoder,
    transformers.models.bert.modeling_bert.BertPooler,
    transformers.models.bert.modeling_bert.BertModel,
    transformers.models.bert.modeling_bert.BertForSequenceClassification,
    transformers.models.albert.modeling_albert.AlbertModel,
    transformers.models.albert.modeling_albert.AlbertTransformer,
    transformers.models.albert.modeling_albert.AlbertLayerGroup,
    transformers.models.albert.modeling_albert.AlbertEmbeddings,
    transformers.models.albert.modeling_albert.AlbertLayer
)


def hook_for(module: nn.Module, summary: list, batch_size: int, convert_fc=False) -> Optional[Callable[[nn.Module, Any, Any], None]]:
    """
    Return the hook, if any, for the given layer type.

    The hook will append a `LayerDescription` to the given summary list when the model containing `module` is executed.

    :param module: a nn.Module to generate a hook for
    :param summary: the summary list we are adding to
    :param batch_size: the input batch size
    :param convert_fc: whether to convert the layer if it is fully connected
    :return: a hook function that can be used with `register_forward_hook()`, or `None` if it does not exist
    """

    if isinstance(module, nn.Linear) and convert_fc:
        return _linear_hook(summary, batch_size)
    elif isinstance(module, nn.Conv2d):
        return _conv_hook(summary, batch_size)
    elif isinstance(module, null_ops):
        # Dropout is not used during inference
        return _null_hook(summary, batch_size)
    elif isinstance(module, nn.LayerNorm):
        if module.elementwise_affine:
            return _layer_norm_hook(summary, batch_size)
    elif isinstance(module, transformers.models.bert.modeling_bert.BertSelfAttention):
        return _multihead_self_attention(summary, batch_size)
    elif isinstance(module, transformers.models.albert.modeling_albert.AlbertAttention):
        return _albert_multihead_self_attention(summary, batch_size)
    elif isinstance(module, transformers.models.albert.modeling_albert.AlbertLayer):
        return _albert_layer(summary, batch_size)
    elif isinstance(module, transformers.models.bert.modeling_bert.BertIntermediate):
        return _null_hook(summary, batch_size)

        #return _bert_intermediate(summary, batch_size)

    print("unknown module type", module.__class__)
