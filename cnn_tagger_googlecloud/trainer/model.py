# Copyright Hongxuan Liu @ 2017. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Define a convolutional neural network for the image classification to identify
a W boson object from the CMS detector data (public data). The detector can
be viewed as a giant camera which takes "photos" of the collisions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dense, Convolution2D, Conv2D, MaxPooling2D, Dropout, Flatten, InputLayer
from keras.utils import np_utils, plot_model
from keras.callbacks import EarlyStopping
import numpy as np
import sys
import glob

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils

import multiprocessing

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from keras.layers.core import K

# Define the image format and number of classes
IMAGE_HEIGHT = 30
IMAGE_WIDTH = 30
CHANNELS = 3
CLASSES = 2

def build_read_and_decode_fn(filenamelist, batch_size, shuffle, compressType = ''):
    """
    Build the read and decode function.
    Inputs:
    filenamelist: a string to where the tfrecord files stored
    compressType: specify if the tfrecord is compressed or not

    Returns:
    A dict of a batch of images and labels. Both images and labels are decoded to the correct
    data type and shapes.
    """
    def _read_and_decode_fn():
        train_filenames = tf.train.match_filenames_once(filenamelist)
        train_filename_queue = tf.train.string_input_producer(train_filenames, num_epochs=None, shuffle=shuffle)
    
        if compressType == 'gzip':
            options = tf.python_io.TFRecordOptions(
                compression_type = tf.python_io.TFRecordCompressionType.GZIP
            )
        elif compressType == 'zlib':
            options = tf.python_io.TFRecordOptions(
                compression_type = tf.python_io.TFRecordCompressionType.ZLIB
            )
        else:
            options = tf.python_io.TFRecordOptions(
                compression_type = tf.python_io.TFRecordCompressionType.NONE
            )

        reader = tf.TFRecordReader(options=options)
        _, serialized_example = reader.read_up_to(train_filename_queue, batch_size)        

        features = tf.parse_example(
            serialized_example,
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string)
            }
        )
    
        image = tf.decode_raw(features['image_raw'], tf.float64)
        label = tf.cast(features['label'], tf.int32)
    
        image_shape = tf.stack([tf.shape(image)[0], IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])
        resized_image = tf.reshape(image, image_shape)
    
        thread_count = multiprocessing.cpu_count()
        min_after_dequeue = 1000
        queue_size_multiplier = thread_count + 3
        if shuffle:
            capacity = min_after_dequeue + queue_size_multiplier * batch_size
            out_image, out_label = tf.train.shuffle_batch(
                                        [resized_image, label], 
                                        batch_size = batch_size, 
                                        capacity = capacity,
                                        min_after_dequeue = min_after_dequeue,
                                        enqueue_many = True,
                                        num_threads = thread_count,
                                        allow_smaller_final_batch = True
                                   )
            return {'image': out_image}, {'label': out_label}
        else:
            capacity = queue_size_multiplier * batch_size
            out_image, out_label = tf.train.batch(
                [resized_image, label],
                batch_size = batch_size,
                capacity = capacity,
                enqueue_many = True,            
                num_threads = thread_count,
                allow_smaller_final_batch = True
            )
            return {'image': out_image}, {'label': out_label}
        
    return _read_and_decode_fn

def build_conv_model(inputs, params):
    """
    Function to compose the Convolutional Neural Network model
    A high level Keras API is used to speed up the network building
    """
    model = Sequential()

    # These are tunable hyper-parameters
    num_filters = params['num_filters']
    dropout_prob = params['dropout_prob']
    dense_layer1_size = params['dense_layer1_size']
    filter_size = params['filter_size']
    pool_size = params['pool_size']
   
    # Output of the first layer shape is (None, HEIGHT, WIDTH, num_filters)
    # Total number of paramers are: num_filters * filter_size * filter_size * CHANNELS + num_filters (bias term)
    model.add(InputLayer(input_tensor=inputs))
    model.add(Conv2D(num_filters, (filter_size, filter_size), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout_prob))
   
    # Output of the second layer shape is (None, HEIGHT/pool_size, WIDTH/pool_size, num_filters)
    # Total number of parameters are: num_filters * filter_size * filter_size * num_filters + num_filters 
    model.add(Conv2D(num_filters, (filter_size, filter_size), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout_prob))

    # Ditto
    model.add(Conv2D(num_filters, (filter_size, filter_size), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout_prob))

    model.add(Flatten())
    model.add(Dropout(dropout_prob))
    model.add(Dense(dense_layer1_size, activation='relu'))
    model.add(Dropout(dropout_prob))

    # Output shape is (None, CLASSES)
    # Number of parameters are: (dense_layer1_size + 1) * CLASSES
    # The loss will be defined in tensorflow Estimator
    model.add(Dense(CLASSES, activation=None))
    
    return model

def model_fn(input_images, input_targets, mode, params):
    """
    Provide configuration for the high level API of tf.contrib.learn.Estimator.
    Use the tf.contrib.learn.Estimator and Experiment to speed up building the training (with distributed training)
    code using tensorflow.
    Inputs:
        input_images and input_targets take the dict out of the build_read_and_decode_fn function.
        params: the hyperparameters to tune the training and the cnn.

    Outputs:
        model_fn_lib.ModelFnOps
    """
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        K.set_learning_phase(1)
    else:
        K.set_learning_phase(0)
        
    features = input_images['image']
    conv_model = build_conv_model(features, params)
    logits = conv_model.output
    
    predictions = tf.nn.softmax(logits, name='softmax_tensor')
    
    if mode == tf.contrib.learn.ModeKeys.INFER:
        return model_fn_lib.ModelFnOps(
            mode=mode,
            predictions=predictions,
        )        
    
    targets = input_targets['label']
    targets = tf.one_hot(targets, CLASSES)

    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(targets,1))    
        
    predictions_dict = {
        'classes': tf.cast(tf.argmax(input=logits, axis=1, name='logits_tensor'), tf.int32),
        'probabilities': predictions,
        'accuracy': tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_tensor')
    }

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits))
    
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params["learning_rate"],
        optimizer="Adam")
    
    eval_metric_ops = {
        "accuracy_eval":
            tf.reduce_mean(tf.cast(correct_prediction, tf.float32)),
    }
    
    return model_fn_lib.ModelFnOps(
      mode=mode,
      predictions=predictions_dict,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops
    )

def predict_input_fn():
    """
    Supplies the input to the model for prediction and serving

    Returns:
        A tuple consisting of 1) a dictionary of tensors whose keys are
        the feature names, and 2) a tensor of target labels if the mode
        is not INFER (and None, otherwise).
    """
    # Add a placeholder for the image input.
    feature_placeholders = {'image': tf.placeholder(tf.float64, [None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])}

    features = feature_placeholders
    
    return input_fn_utils.InputFnOps(
          features=features, labels=None, default_inputs=feature_placeholders)
