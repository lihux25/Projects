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
Serving the trained network locally
"""

import argparse

import model

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils

from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.contrib.session_bundle import manifest_pb2

from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
import glob
import numpy as np

import matplotlib.pyplot as plt

def decode_to_np_arrays(tfrecord_file_name, compressType = ''):
    label_list = []
    img_list = []
    
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
    for file_name in tfrecord_file_name:
        record_iterator = tf.python_io.tf_record_iterator(path=file_name, options=options)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            height = int(example.features.feature['height'].int64_list.value[0])
            width = int(example.features.feature['width'].int64_list.value[0])
            depth = int(example.features.feature['depth'].int64_list.value[0])
            label = int(example.features.feature['label'].int64_list.value[0])
            img_string = (example.features.feature['image_raw'].bytes_list.value[0])
    
            img_1d = np.fromstring(img_string, dtype=np.float)
            reco_img = img_1d.reshape((height, width, depth))
    
            label_list.append(label)
            img_list.append(reco_img)

    reco_images={'label':np.array(label_list), 'data':np.array(img_list)}
    return reco_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--model-dir',
        help='location to load the serving models',
        required=True
    )
    parser.add_argument(
        '--input-list',
        help='input tfrecord file names',
        required=True
    )
    parser.add_argument(
        '--test-n-samples',
        help='number of samples for testing',
        default = -1,
        type=int
    )

    # Argument to turn on all logging
    parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default=tf.logging.INFO,
        help='Set logging verbosity'
    )

    args = parser.parse_args()
    arguments = args.__dict__
    tf.logging.set_verbosity(arguments.pop('verbosity'))

    model_dir = arguments.pop('model_dir')
    input_list = arguments.pop('input_list')
    test_n_samples = arguments.pop('test_n_samples')

    full_input_filenames = glob.glob(input_list)

    decode_test = decode_to_np_arrays(full_input_filenames, 'gzip')
    jet_images_test = decode_test['data']
    labels_test = decode_test['label'] 

    encoder = LabelEncoder()
    encoder.fit(labels_test)
    encoded_labels_test_trans = encoder.transform(labels_test)
    # Transform to one-hot format
    encoded_labels_test = np_utils.to_categorical(encoded_labels_test_trans)
    
    feed_jet_images_test = jet_images_test if test_n_samples <0 else jet_images_test[:test_n_samples]
    feed_labels_test = labels_test if test_n_samples <0 else labels_test[:test_n_samples]

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_dir)

#        all_keys = sess.graph.get_all_collection_keys()
#        print(all_keys)

#        print([v.op.name for v in tf.global_variables()])
#        print([n.name for n in sess.graph.as_graph_def().node])

        print([n.name + '=>' +  n.op for n in sess.graph.as_graph_def().node if n.op in ('Softmax','Placeholder')])

        softmax_tensor = sess.graph.get_tensor_by_name('softmax_tensor:0')

        probs = sess.run(softmax_tensor, {'Placeholder:0' : feed_jet_images_test})

        fpr, tpr, thresholds = roc_curve(feed_labels_test, probs[:,1])
        roc_auc = auc(fpr, tpr)

        color = 'blue'
        lw = 2

        fig = plt.figure()

        plt.plot(fpr, tpr, lw=lw, color=color, label='ROC area = %0.2f' % (roc_auc))

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        fig.savefig('roc.png', bbox_inches='tight')

        print(roc_auc)
