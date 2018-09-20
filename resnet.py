from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-3
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

# Convenience functions for building the ResNet model

def batch_norm(inputs, training, data_format):
    return tf.layers.batch_normalization(
        inputs = inputs, axis = 1 if data_format == 'channels_first' else 3,
        momentum = _BATCH_NORM_DECAY, epsilon = _BATCH_NORM_EPSILON,
        training = training)

def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size."""

    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):

    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    
    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(scale=2.0),
        data_format=data_format)

def _regular_block_v1(inputs, filters, training, projection_shortcut,
                      strides, data_format):
    shortcut = inputs
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training,
                              data_format=data_format)
        
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3,
                                  strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3,
                                  strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)
    return inputs

def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                         strides, data_format):

    """A single block for ResNet v1, with a bottleneck.

    Args:

    """
    # same as tensorflow implementation?
    shortcut = inputs
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training,
                              data_format=data_format)


    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1,
                                  strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3,
                                  strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters*4, kernel_size=1,
                                  strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs

def block_layer(inputs, filters, num_blocks, strides, training, bottleneck, name, data_format):
    if bottleneck:
        block = _bottleneck_block_v1
        out_filters = filters * 4
    else:
        block = _regular_block_v1
        out_filters = filters
    projection_shortcut = None
    
    if strides != 1:
        def projection_shortcut(inputs):
            return conv2d_fixed_padding(
                inputs=inputs, filters=out_filters, kernel_size=1, strides=strides,
                data_format=data_format)


    inputs = block(inputs, filters, training, projection_shortcut, strides,
                                  data_format)

    for _ in range(1, num_blocks):
        inputs = block(inputs, filters, training, None, 1, data_format)

    return tf.identity(inputs, name)

# how do inputs when using tf.data.Datasets work? (do not use tf.placeholder, correct?)
def resnet(batch, num_classes, training, init_kernel_size=7, block_sizes=[3, 4, 6, 3],
           init_num_filters = 64, init_conv_stride = 2, init_pool_size = 3,
           inti_pool_stride = 2, bottleneck = True, data_format = 'channels_first'):
    # is this necessary when using tf.layers or get_variables?
    # I guess I do not understand its utility yet
    if data_format == 'channels_first':
        inputs = tf.transpose(batch, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=init_num_filters, kernel_size=init_kernel_size,
        strides=init_conv_stride, data_format=data_format)
        
    # Best way to name an operation? Alternatives?
    inputs = tf.identity(inputs, 'initial_conv')

    # how do I alternate between training and evaluating?
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    if init_pool_size:
        inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=init_pool_size,
                                             strides=init_pool_stride, padding='SAME',
                                             data_format=data_format)

    # how does this appear on the graph? is it like annotation?
    inputs = tf.identity(inputs, 'initial_max_pool')

    for i, num_blocks in enumerate(block_sizes):
        # double the number of filters after each block
        strides = 1 if i == 0 else 2

        num_filters = init_num_filters * (2**i)
        inputs = block_layer(inputs=inputs, filters=num_filters,
                             num_blocks=num_blocks, strides=strides, training=training,
                             bottleneck=bottleneck, name=f'block_layer{i+1}',
                             data_format=data_format)

    axes = [2, 3] if data_format == 'channels_first' else [1, 2]
    inputs = tf.reduce_mean(inputs, axes, keepdims=True)
    inputs = tf.identity(inputs, 'final_reduce_mean')
    
    inputs = tf.squeeze(inputs, axes)
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')
    return inputs
            
