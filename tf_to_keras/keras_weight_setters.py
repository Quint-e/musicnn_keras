import tensorflow as tf

def set_vgg_weights(model,model_vars):
    # print(model.get_weights())
    # print(model.layers)
    keras_layer_names = [{'name':'bn_input','type':'batchnorm'},
                         {'name':'1CNN','type':'conv2d'}, {'name':'bn_conv1','type':'batchnorm'},
                         {'name':'2CNN','type':'conv2d'}, {'name':'bn_conv2','type':'batchnorm'},
                         {'name':'3CNN','type':'conv2d'}, {'name':'bn_conv3','type':'batchnorm'},
                         {'name':'4CNN','type':'conv2d'},  {'name':'bn_conv4','type':'batchnorm'},
                         {'name':'5CNN','type':'conv2d'}, {'name':'bn_conv5','type':'batchnorm'},
                         {'name':'output','type':'dense'}]
    tf_layer_names = ['batch_normalization',
                      '1CNN','batch_normalization_1',
                      '2CNN','batch_normalization_2',
                      '3CNN','batch_normalization_3',
                      '4CNN','batch_normalization_4',
                      '5CNN','batch_normalization_5',
                      'dense']

    # # printing routine
    # for i, layer in enumerate(model.layers):
    #     name = layer._name
    #     for weight_array in layer.get_weights():
    #         print(name, weight_array)


    for i, keras_layer_dict in enumerate(keras_layer_names):
        keras_layer_name = keras_layer_dict['name']
        layer_type = keras_layer_dict['type']
        layer = model.get_layer(keras_layer_name)
        tf_layer_name = tf_layer_names[i]
        if layer_type == 'batchnorm':
            layer.set_weights([model_vars['{}/gamma:0'.format(tf_layer_name)],
                               model_vars['{}/beta:0'.format(tf_layer_name)],
                               model_vars['{}/moving_mean:0'.format(tf_layer_name)],
                               model_vars['{}/moving_variance:0'.format(tf_layer_name)]])
        elif layer_type == 'conv2d':
            layer.set_weights([model_vars['{}/kernel:0'.format(tf_layer_name)],
                               model_vars['{}/bias:0'.format(tf_layer_name)]])
        elif layer_type == 'dense':
            layer.set_weights([model_vars['{}/kernel:0'.format(tf_layer_name)],
                               model_vars['{}/bias:0'.format(tf_layer_name)]])
    return model


def set_musicnn_weights(model,model_vars):
    # print(model.get_weights())
    # print(model.layers)
    keras_layer_names = [{'name':'batch_normalization','type':'batchnorm'},
                         {'name':'conv2d','type':'conv2d'},   {'name':'batch_normalization_1','type':'batchnorm'},
                         {'name':'conv2d_1','type':'conv2d'}, {'name':'batch_normalization_2','type':'batchnorm'},
                         {'name':'conv2d_2','type':'conv2d'}, {'name':'batch_normalization_3','type':'batchnorm'},
                         {'name':'conv2d_3','type':'conv2d'}, {'name':'batch_normalization_4','type':'batchnorm'},
                         {'name':'conv2d_4','type':'conv2d'}, {'name':'batch_normalization_5','type':'batchnorm'},
                         {'name': 'midend_conv1', 'type': 'conv2d'}, {'name': 'bn_midend_conv1', 'type': 'batchnorm'},
                         {'name': 'midend_conv2', 'type': 'conv2d'}, {'name': 'bn_midend_conv2', 'type': 'batchnorm'},
                         {'name': 'midend_conv3', 'type': 'conv2d'}, {'name': 'bn_midend_conv3', 'type': 'batchnorm'},
                         {'name': 'bn_flatpool', 'type': 'batchnorm'},
                         {'name': 'dense', 'type': 'dense'},
                         {'name': 'bn_dense', 'type': 'batchnorm'},
                         {'name':'dense_1','type':'dense'}]
    tf_layer_names = ['batch_normalization',
                      'conv2d','batch_normalization_1',
                      'conv2d_1', 'batch_normalization_2',
                      'conv2d_2', 'batch_normalization_3',
                      'conv2d_3', 'batch_normalization_4',
                      'conv2d_4', 'batch_normalization_5',
                      'conv2d_5', 'batch_normalization_6',
                      'conv2d_6', 'batch_normalization_7',
                      'conv2d_7', 'batch_normalization_8',
                      'batch_normalization_9',
                      'dense',
                      'batch_normalization_10',
                      'dense_1']

    # # printing routine
    # for i, layer in enumerate(model.layers):
    #     name = layer._name
    #     for weight_array in layer.get_weights():
    #         print(name, weight_array)


    for i, keras_layer_dict in enumerate(keras_layer_names):
        keras_layer_name = keras_layer_dict['name']
        layer_type = keras_layer_dict['type']
        layer = model.get_layer(keras_layer_name)
        tf_layer_name = tf_layer_names[i]
        if layer_type == 'batchnorm':
            layer.set_weights([model_vars['{}/gamma:0'.format(tf_layer_name)],
                               model_vars['{}/beta:0'.format(tf_layer_name)],
                               model_vars['{}/moving_mean:0'.format(tf_layer_name)],
                               model_vars['{}/moving_variance:0'.format(tf_layer_name)]])
        elif layer_type == 'conv2d':
            layer.set_weights([model_vars['{}/kernel:0'.format(tf_layer_name)],
                               model_vars['{}/bias:0'.format(tf_layer_name)]])
        elif layer_type == 'dense':
            layer.set_weights([model_vars['{}/kernel:0'.format(tf_layer_name)],
                               model_vars['{}/bias:0'.format(tf_layer_name)]])
    return model