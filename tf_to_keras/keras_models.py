# -*- coding: utf-8 -*-
"""
@author: quintoe
python3.6

"""
import sys
sys.path.insert(0,'../../musicnn_keras')
import os
import numpy as np
import tensorflow as tf
from musicnn_keras import configuration as config

################################## VGG Model ####################################
def vgg(num_classes, num_filters=32):
    # input_expanded = tf.expand_dims(x, 3) #Adding channel dimension I guess...

    input_shape = (187,96,1)

    input_layer = tf.keras.Input(shape=input_shape)
    bn_input = tf.keras.layers.BatchNormalization(axis=-1, name='bn_input',
                                                  beta_initializer='zeros', gamma_initializer='ones',
                                                  moving_mean_initializer=tf.constant_initializer(value=0.25),
                                                  moving_variance_initializer=tf.constant_initializer(value=0.75)
                                                  )(input_layer)

    conv1 = tf.keras.layers.Conv2D(filters=num_filters,
                                   kernel_size=[3, 3],
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='1CNN')(bn_input)
    bn_conv1 = tf.keras.layers.BatchNormalization(axis=-1, name='bn_conv1')(conv1)
    pool1 = tf.keras.layers.MaxPool2D(pool_size=[4, 1], strides=[2, 2],name='pool1')(bn_conv1)

    do_pool1 = tf.keras.layers.Dropout(rate=0.25)(pool1)
    conv2 = tf.keras.layers.Conv2D(filters=num_filters,
                                   kernel_size=[3, 3],
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='2CNN')(do_pool1)
    bn_conv2 = tf.keras.layers.BatchNormalization(axis=-1, name='bn_conv2')(conv2)
    pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], name='pool2')(bn_conv2)

    do_pool2 = tf.keras.layers.Dropout(rate=0.25)(pool2)
    conv3 = tf.keras.layers.Conv2D(filters=num_filters,
                                   kernel_size=[3, 3],
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='3CNN')(do_pool2)

    bn_conv3 = tf.keras.layers.BatchNormalization(axis=-1, name='bn_conv3')(conv3)
    pool3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], name='pool3')(bn_conv3)

    do_pool3 = tf.keras.layers.Dropout(rate=0.25)(pool3)
    conv4 = tf.keras.layers.Conv2D(filters=num_filters,
                                   kernel_size=[3, 3],
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='4CNN')(do_pool3)

    bn_conv4 = tf.keras.layers.BatchNormalization(axis=-1, name='bn_conv4')(conv4)
    pool4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], name='pool4')(bn_conv4)

    do_pool4 = tf.keras.layers.Dropout(rate=0.25)(pool4)
    conv5 = tf.keras.layers.Conv2D(filters=num_filters,
                                   kernel_size=[3, 3],
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='5CNN')(do_pool4)

    bn_conv5 = tf.keras.layers.BatchNormalization(axis=-1, name='bn_conv5')(conv5)
    pool5 = tf.keras.layers.MaxPool2D(pool_size=[4, 4], strides=[4, 4], name='pool5')(bn_conv5)

    flat_pool5 = tf.keras.layers.Flatten()(pool5)
    do_pool5 = tf.keras.layers.Dropout(rate=0.5)(flat_pool5)
    output = tf.keras.layers.Dense(num_classes, activation='linear', name='output')(do_pool5)

    # Create model
    model = tf.keras.Model(input_layer, output)
    return model

######################################################################################################


############################################ Music CNN Model ###########################################


def build_musicnn(num_classes, num_filt_frontend=1.6, num_filt_midend=64, num_units_backend=200):
    '''For musiccnn_big, set num_filt_midend=512 and num_units_backend=500'''

    input_shape = (187, 96, 1)
    input_layer = tf.keras.Input(shape=input_shape)

    ### front-end ### musically motivated CNN
    frontend_features_list = frontend(input_layer, config.N_MELS, num_filt=num_filt_frontend, type='7774timbraltemporal')
    # concatnate features coming from the front-end
    frontend_features = tf.concat(frontend_features_list, 2)

    ### mid-end ### dense layers
    midend_features_list = midend(frontend_features, num_filt_midend)
    # dense connection: concatnate features coming from different layers of the front- and mid-end
    midend_features = tf.concat(midend_features_list, 2)

    ### back-end ### temporal pooling
    logits, penultimate, mean_pool, max_pool = backend(midend_features, num_classes, num_units_backend, type='globalpool_dense')

    # # [extract features] temporal and timbral features from the front-end
    # timbral = tf.concat([frontend_features_list[0], frontend_features_list[1]], 2)
    # temporal = tf.concat([frontend_features_list[2], frontend_features_list[3], frontend_features_list[4]], 2)
    # # [extract features] mid-end features
    # cnn1, cnn2, cnn3 = midend_features_list[1], midend_features_list[2], midend_features_list[3]
    # mean_pool = tf.squeeze(mean_pool, [2])
    # max_pool = tf.squeeze(max_pool, [2])
    #
    # # return logits, timbral, temporal, cnn1, cnn2, cnn3, mean_pool, max_pool, penultimate

    output = tf.keras.layers.Activation('sigmoid')(logits)

    return tf.keras.Model(input_layer,output)


def timbral_block(inputs, filters, kernel_size, padding="valid", activation=tf.nn.relu):

    conv = tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  activation=activation
                                  )(inputs)
    bn_conv = tf.keras.layers.BatchNormalization(axis=-1)(conv)
    pool = tf.keras.layers.MaxPool2D(pool_size=[1, bn_conv.shape[2]],
                                     strides=[1, bn_conv.shape[2]]
                                     )(bn_conv)
    return tf.squeeze(pool, [2])


def tempo_block(inputs, filters, kernel_size, padding="same", activation=tf.nn.relu):

    conv = tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  activation=activation
                                  )(inputs)
    bn_conv = tf.keras.layers.BatchNormalization(axis=-1)(conv)
    pool = tf.keras.layers.MaxPool2D(pool_size=[1, bn_conv.shape[2]],
                                     strides=[1, bn_conv.shape[2]])(bn_conv)
    return tf.squeeze(pool, [2])

def frontend(input_layer, yInput, num_filt, type):
    '''yInput is the number of mel bins in practice. '''

    # expand_input = tf.expand_dims(x, 3) #Not necessary because should be done earlier.
    normalized_input = tf.keras.layers.BatchNormalization(axis=-1)(input_layer)

    if 'timbral' in type:

        # padding only time domain for an efficient 'same' implementation
        # (since we pool throughout all frequency afterwards)
        input_pad_7 = tf.pad(normalized_input, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")

        if '74' in type:
            f74 = timbral_block(inputs=input_pad_7,
                                filters=int(num_filt*128),
                                kernel_size=[7, int(0.4 * yInput)]
                                )

        if '77' in type:
            f77 = timbral_block(inputs=input_pad_7,
                                filters=int(num_filt*128),
                                kernel_size=[7, int(0.7 * yInput)]
                                )

    if 'temporal' in type:

        s1 = tempo_block(inputs=normalized_input,
                          filters=int(num_filt*32),
                          kernel_size=[128,1]
                         )

        s2 = tempo_block(inputs=normalized_input,
                          filters=int(num_filt*32),
                          kernel_size=[64,1]
                         )

        s3 = tempo_block(inputs=normalized_input,
                          filters=int(num_filt*32),
                          kernel_size=[32,1]
                         )

    # choose the feature maps we want to use for the experiment
    if type == '7774timbraltemporal':
        return [f74, f77, s1, s2, s3]


def midend(front_end_output, num_filt):

    front_end_output = tf.expand_dims(front_end_output, 3)

    # conv layer 1 - adapting dimensions
    front_end_pad = tf.pad(front_end_output, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv1 = tf.keras.layers.Conv2D(filters=num_filt,
                                   kernel_size=[7, front_end_pad.shape[2]],
                                   activation=tf.nn.relu,
                                   name='midend_conv1')(front_end_pad)
    bn_conv1 = tf.keras.layers.BatchNormalization(axis=-1,name='bn_midend_conv1')(conv1)
    bn_conv1_t = tf.transpose(bn_conv1, [0, 1, 3, 2])

    # conv layer 2 - residual connection
    bn_conv1_pad = tf.pad(bn_conv1_t, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv2 = tf.keras.layers.Conv2D(filters=num_filt,
                                   kernel_size=[7, bn_conv1_pad.shape[2]],
                                   padding="valid",
                                   activation=tf.nn.relu,
                                   name='midend_conv2')(bn_conv1_pad)
    bn_conv2 = tf.keras.layers.BatchNormalization(axis=-1,name='bn_midend_conv2')(conv2)
    conv2 = tf.transpose(bn_conv2, [0, 1, 3, 2])
    res_conv2 = tf.add(conv2, bn_conv1_t)

    # conv layer 3 - residual connection
    bn_conv2_pad = tf.pad(res_conv2, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv3 = tf.keras.layers.Conv2D(filters=num_filt,
                                   kernel_size=[7, bn_conv2_pad.shape[2]],
                                   padding="valid",
                                   activation=tf.nn.relu,
                                   name='midend_conv3')(bn_conv2_pad)
    bn_conv3 = tf.keras.layers.BatchNormalization(axis=-1,name='bn_midend_conv3')(conv3)
    conv3 = tf.transpose(bn_conv3, [0, 1, 3, 2])
    res_conv3 = tf.add(conv3, res_conv2)

    return [front_end_output, bn_conv1_t, res_conv2, res_conv3]



def backend(feature_map, num_classes, output_units, type):

    # temporal pooling
    max_pool = tf.reduce_max(feature_map, axis=1)
    mean_pool, var_pool = tf.nn.moments(feature_map, axes=[1])
    tmp_pool = tf.concat([max_pool, mean_pool], 2)

    # penultimate dense layer
    flat_pool = tf.keras.layers.Flatten()(tmp_pool)
    flat_pool = tf.keras.layers.BatchNormalization(axis=-1,name='bn_flatpool')(flat_pool)
    flat_pool_dropout = tf.keras.layers.Dropout(rate=0.5)(flat_pool)
    dense = tf.keras.layers.Dense(units=output_units,
                                  activation=tf.nn.relu)(flat_pool_dropout)
    bn_dense = tf.keras.layers.BatchNormalization(axis=-1,name='bn_dense')(dense)
    dense_dropout = tf.keras.layers.Dropout(rate=0.5)(bn_dense)

    # output dense layer
    logits = tf.keras.layers.Dense(activation=None,
                                   units=num_classes)(dense_dropout)

    return logits, bn_dense, mean_pool, max_pool

###############################################################################################
