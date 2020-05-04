import os
import numpy as np
import librosa
import tensorflow as tf
import sys
sys.path.insert(0,'../../musicnn_keras')
from keras_models import vgg, build_musicnn
from keras_weight_setters import set_vgg_weights, set_musicnn_weights
from musicnn_keras import configuration as config

def load_tf_checkpoint(path_meta):
    # start tensorflow session
    with tf.compat.v1.Session() as sess:

        # import graph
        saver = tf.compat.v1.train.import_meta_graph(path_meta)

        # load weights for graph
        saver.restore(sess, path_meta[:-5])

        # get all global variables (including model variables)
        vars_global = tf.compat.v1.global_variables()

        # get their name and value and put them into dictionary
        sess.as_default()
        model_vars = {}
        for var in vars_global:
            try:
                model_vars[var.name] = var.eval()
            except:
                print("For var={}, an exception occurred".format(var.name))
    return model_vars

def load_vgg_keras(num_classes):
    vgg_model = vgg(num_classes, num_filters=128)
    return vgg_model

def load_musicnn_keras(num_classes):
    musicnn_model = build_musicnn(num_classes, num_filt_frontend=1.6, num_filt_midend=64, num_units_backend=200)
    return musicnn_model

def load_musicnn_big_keras(num_classes):
    musicnn_big_model = build_musicnn(num_classes, num_filt_frontend=1.6, num_filt_midend=512, num_units_backend=500)
    return musicnn_big_model

def compile_model(model_keras):
    # Compile model
    opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
    model_keras.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model_keras.summary()
    return model_keras

# disable eager mode for tf.v1 compatibility with tf.v2
tf.compat.v1.disable_eager_execution()

# TF checkpoint filepaths. Uncomment one at a time to generate the equivalent tf.keras model.
# PATH_REL_META = r'./musicnn/MTT_vgg/.meta'
# PATH_REL_META = r'./musicnn/MSD_vgg/.meta'
PATH_REL_META = r'./musicnn/MSD_musicnn/.meta'
# PATH_REL_META = r'./musicnn/MSD_musicnn_big/.meta'
# PATH_REL_META = r'./musicnn/MTT_musicnn/.meta'

models_keras_dir = '../musicnn_keras/keras_checkpoints/'

model_name = os.path.split(os.path.split(PATH_REL_META)[0])[1]
# Load TF checkpoint
model_vars = load_tf_checkpoint(PATH_REL_META)

# Load keras model
if  model_name=='MSD_vgg':
    num_classes = len(config.MSD_LABELS)
    model_keras = load_vgg_keras(num_classes)
    model_keras = compile_model(model_keras)
    # Set weights
    model_keras = set_vgg_weights(model_keras, model_vars)
elif  model_name=='MTT_vgg':
    num_classes = len(config.MTT_LABELS)
    model_keras = load_vgg_keras(num_classes)
    model_keras = compile_model(model_keras)
    # Set weights
    model_keras = set_vgg_weights(model_keras, model_vars)
elif model_name == 'MSD_musicnn':
    num_classes = len(config.MSD_LABELS)
    model_keras = load_musicnn_keras(num_classes)
    model_keras = compile_model(model_keras)
    # Set weights
    model_keras = set_musicnn_weights(model_keras,model_vars)
elif model_name == 'MSD_musicnn_big':
    num_classes = len(config.MSD_LABELS)
    model_keras = load_musicnn_big_keras(num_classes)
    model_keras = compile_model(model_keras)
    # Set weights
    model_keras = set_musicnn_weights(model_keras,model_vars)
elif model_name == 'MTT_musicnn':
    num_classes = len(config.MTT_LABELS)
    model_keras = load_musicnn_keras(num_classes)
    model_keras = compile_model(model_keras)
    # Set weights
    model_keras = set_musicnn_weights(model_keras,model_vars)

    # Save keras model
    model_path = os.path.join(models_keras_dir,'{}.h5'.format(model_name))
    model_keras.save(model_path)