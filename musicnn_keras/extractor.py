import os
import numpy as np
import librosa

import tensorflow as tf
from musicnn_keras import configuration as config


def batch_data(audio_file, n_frames, overlap):
    '''For an efficient computation, we split the full music spectrograms in patches of length n_frames with overlap.

    INPUT
    
    - file_name: path to the music file to tag.
    Data format: string.
    Example: './audio/TRWJAZW128F42760DD_test.mp3'

    - n_frames: length (in frames) of the input spectrogram patches.
    Data format: integer.
    Example: 187
        
    - overlap: ammount of overlap (in frames) of the input spectrogram patches.
    Note: Set it considering n_frames.
    Data format: integer.
    Example: 10
    
    OUTPUT
    
    - batch: batched audio representation. It returns spectrograms split in patches of length n_frames with overlap.
    Data format: 3D np.array (batch, time, frequency)
    
    - audio_rep: raw audio representation (spectrogram).
    Data format: 2D np.array (time, frequency)
    '''

    # compute the log-mel spectrogram with librosa
    audio, sr = librosa.load(audio_file, sr=config.SR)
    audio_rep = librosa.feature.melspectrogram(y=audio, 
                                               sr=sr,
                                               hop_length=config.FFT_HOP,
                                               n_fft=config.FFT_SIZE,
                                               n_mels=config.N_MELS).T
    audio_rep = audio_rep.astype(np.float16)
    audio_rep = np.log10(10000 * audio_rep + 1)

    # batch it for an efficient computing
    first = True
    last_frame = audio_rep.shape[0] - n_frames + 1
    # +1 is to include the last frame that range would not include
    for time_stamp in range(0, last_frame, overlap):
        patch = np.expand_dims(audio_rep[time_stamp : time_stamp + n_frames, : ], axis=0)
        if first:
            batch = patch
            first = False
        else:
            batch = np.concatenate((batch, patch), axis=0)

    return batch, audio_rep


def extractor(file_name, model='MSD_musicnn', input_length=3, input_overlap=False, extract_features=False):
    '''Extract the taggram (the temporal evolution of tags) and features (intermediate representations of the model) of the music-clip in file_name with the selected model.

    INPUT

    - file_name: path to the music file to tag.
    Data format: string.
    Example: './audio/TRWJAZW128F42760DD_test.mp3'
    
    - model: select a music audio tagging model.
    Data format: string.
    Options: 'MTT_musicnn', 'MTT_vgg', 'MSD_musicnn', 'MSD_musicnn_big' or 'MSD_vgg'.
    MTT models are trained with the MagnaTagATune dataset.
    MSD models are trained with the Million Song Dataset.
    To know more about these models, check our musicnn / vgg examples, and the FAQs.
    Important! 'MSD_musicnn_big' is only available if you install from source: python setup.py install.

    - input_length: length (in seconds) of the input spectrogram patches. Set it small for real-time applications.
    Note: This is the length of the data that is going to be fed to the model. In other words, this parameter defines the temporal resolution of the taggram.
    Recommended value: 3, because the models were trained with 3 second inputs.
    Observation: the vgg models do not allow for different input lengths. For this reason, the vgg models' input_length needs to be set to 3. However, musicnn models allow for different input lengths: see this jupyter notebook.
    Data format: floating point number.
    Example: 3.1
    
    - input_overlap: ammount of overlap (in seconds) of the input spectrogram patches.
    Note: Set it considering the input_length.
    Data format: floating point number.
    Example: 1.0
    
    - extract_features: set it True for extracting the intermediate representations of the model.
    Data format: boolean.
    Options: False (for NOT extracting the features), True (for extracting the features).

    OUTPUT
    
    - taggram: expresses the temporal evolution of the tags likelihood.
    Data format: 2D np.ndarray (time, tags).
    Example: see our basic / advanced examples.
    
    - tags: list of tags corresponding to the tag-indices of the taggram.
    Data format: list.
    Example: see our FAQs page for the complete tags list.
    
    - features: if extract_features = True, it outputs a dictionary containing the activations of the different layers the selected model has.
    Data format: dictionary.
    Keys (musicnn models): ['timbral', 'temporal', 'cnn1', 'cnn2', 'cnn3', 'mean_pool', 'max_pool', 'penultimate']
    Keys (vgg models): ['pool1', 'pool2', 'pool3', 'pool4', 'pool5']
    Example: see our musicnn and vgg examples.

    '''
    # loading tf.keras model
    try:
        keras_model = tf.keras.models.load_model('./musicnn_keras/keras_checkpoints/{}.h5'.format(model))
    except:
        raise ValueError('Unknown model')

    # select labels
    if 'MTT' in model:
        labels = config.MTT_LABELS
    elif 'MSD' in model:
        labels = config.MSD_LABELS
    
    if 'vgg' in model and input_length != 3:
        raise ValueError('Set input_length=3, the VGG models cannot handle different input lengths.')

    # convert seconds to frames
    n_frames = librosa.time_to_frames(input_length, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP) + 1
    if not input_overlap:
        overlap = n_frames
    else:
        overlap = librosa.time_to_frames(input_overlap, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP)



    # batching data
    print('Computing spectrogram (w/ librosa) and tags (w/ tensorflow)..', end =" ")
    batch, spectrogram = batch_data(file_name, n_frames, overlap)
    batch = tf.expand_dims(batch, 3)  # Add channel dimension

    # extract tagram
    taggram = keras_model.predict_on_batch(batch)

    if extract_features:
        if 'vgg' in model:
            intermediate_layer_model_pool1 = tf.keras.Model(keras_model.input,
                                                            keras_model.get_layer('pool1').output)
            intermediate_layer_model_pool2 = tf.keras.Model(keras_model.input,
                                                            keras_model.get_layer('pool2').output)
            intermediate_layer_model_pool3 = tf.keras.Model(keras_model.input,
                                                            keras_model.get_layer('pool3').output)
            intermediate_layer_model_pool4 = tf.keras.Model(keras_model.input,
                                                            keras_model.get_layer('pool4').output)
            intermediate_layer_model_pool5 = tf.keras.Model(keras_model.input,
                                                            keras_model.get_layer('pool5').output)

            pool1_out = np.squeeze(intermediate_layer_model_pool1.predict_on_batch(batch))
            pool2_out = np.squeeze(intermediate_layer_model_pool2.predict_on_batch(batch))
            pool3_out = np.squeeze(intermediate_layer_model_pool3.predict_on_batch(batch))
            pool4_out = np.squeeze(intermediate_layer_model_pool4.predict_on_batch(batch))
            pool5_out = np.squeeze(intermediate_layer_model_pool5.predict_on_batch(batch))

            features = dict()
            features['pool1'] = np.reshape(pool1_out, (pool1_out.shape[0]*pool1_out.shape[1],pool1_out.shape[2],pool1_out.shape[3]))
            features['pool2'] = np.reshape(pool2_out, (pool2_out.shape[0]*pool2_out.shape[1],pool2_out.shape[2],pool2_out.shape[3]))
            features['pool3'] = np.reshape(pool3_out, (pool3_out.shape[0]*pool3_out.shape[1],pool3_out.shape[2],pool3_out.shape[3]))
            features['pool4'] = np.reshape(pool4_out, (pool4_out.shape[0]*pool4_out.shape[1],pool4_out.shape[2],pool4_out.shape[3]))
            features['pool5'] = np.reshape(pool5_out, (pool5_out.shape[0]*pool5_out.shape[1],pool5_out.shape[2]))
        else:
            intermediate_layer_model_frontend_features = tf.keras.Model(keras_model.input,
                                                               keras_model.get_layer('tf_op_layer_concat').output)
            intermediate_layer_model_cnn1 = tf.keras.Model(keras_model.input,
                                                            keras_model.get_layer('tf_op_layer_transpose').output)
            intermediate_layer_model_cnn2 = tf.keras.Model(keras_model.input,
                                                           keras_model.get_layer('tf_op_layer_transpose_1').output)
            intermediate_layer_model_cnn3 = tf.keras.Model(keras_model.input,
                                                           keras_model.get_layer('tf_op_layer_transpose_2').output)
            intermediate_layer_model_meanpool = tf.keras.Model(keras_model.input,
                                                           keras_model.get_layer('tf_op_layer_moments/Squeeze').output)
            intermediate_layer_model_maxpool = tf.keras.Model(keras_model.input,
                                                           keras_model.get_layer('tf_op_layer_Max').output)
            intermediate_layer_model_penultimate = tf.keras.Model(keras_model.input,
                                                                keras_model.get_layer('bn_dense').output)

            features_out = np.squeeze(intermediate_layer_model_frontend_features.predict_on_batch(batch))
            cnn_1_out = np.squeeze(intermediate_layer_model_cnn1.predict_on_batch(batch))
            cnn_2_out = np.squeeze(intermediate_layer_model_cnn2.predict_on_batch(batch))
            cnn_3_out = np.squeeze(intermediate_layer_model_cnn3.predict_on_batch(batch))
            meanpool_out = np.squeeze(intermediate_layer_model_meanpool.predict_on_batch(batch))
            maxpool_out = np.squeeze(intermediate_layer_model_maxpool.predict_on_batch(batch))
            penultimate_out = np.squeeze(intermediate_layer_model_penultimate.predict_on_batch(batch))

            timbral_out  = features_out[:,:,:408]
            temporal_out = features_out[:,:,408:]

            features = dict()
            features['timbral'] = np.reshape(timbral_out, (timbral_out.shape[0]*timbral_out.shape[1],timbral_out.shape[2]))
            features['temporal'] = np.reshape(temporal_out, (temporal_out.shape[0]*temporal_out.shape[1],temporal_out.shape[2]))
            features['cnn1'] = np.reshape(cnn_1_out, (cnn_1_out.shape[0]*cnn_1_out.shape[1],cnn_1_out.shape[2]))
            features['cnn2'] = np.reshape(cnn_2_out, (cnn_2_out.shape[0]*cnn_2_out.shape[1],cnn_2_out.shape[2]))
            features['cnn3'] = np.reshape(cnn_3_out, (cnn_3_out.shape[0]*cnn_3_out.shape[1],cnn_3_out.shape[2]))
            features['mean_pool'] = meanpool_out
            features['max_pool'] = maxpool_out
            features['penultimate'] = penultimate_out


    if extract_features:
        return taggram, labels, features
    else:
        return taggram, labels


