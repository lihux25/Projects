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
Define the task
"""

import argparse

import model

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils

def generate_experiment_fn(train_file_pattern, 
                           eval_file_pattern, 
                           batch_size,
                           shuffle,
                           tfrecord_compress_type,
                           check_n_iter,
                           num_epochs,
                           eval_steps,
                           model_params,
                           **experiment_args):
    """Create an experiment function given hyperparameters.

    See command line help text for description of args.
    Returns:
        A function (output_dir) -> Experiment where output_dir is a string
        representing the location of summaries, checkpoints, and exports.
        this function is used by learn_runner to create an Experiment which
        executes model code provided in the form of an Estimator and
        input functions.

        All listed arguments in the outer function are used to create an
        Estimator, and input functions (training, evaluation, serving).
        Unlisted args are passed through to Experiment.
    """
    def _experiment_fn(output_dir):

        train_input_fn = model.build_read_and_decode_fn(train_file_pattern, batch_size, shuffle, tfrecord_compress_type)
        eval_input_fn = model.build_read_and_decode_fn(eval_file_pattern, batch_size, shuffle, tfrecord_compress_type)

        # The tensors will be printed to the log, with INFO severity.
        tensors_to_log = {'accuracy_train': 'accuracy_tensor'}
        logging_hook = tf.train.LoggingTensorHook(
                                tensors=tensors_to_log, every_n_iter=check_n_iter)

        eval_metric_add = {
            'training/hptuning/metric':  tf.contrib.learn.MetricSpec(
                            metric_fn=tf.contrib.metrics.streaming_accuracy,
                            prediction_key="classes"
                        ),
        }
    
        estimator = tf.contrib.learn.Estimator(
            model_fn=model.model_fn,
            params=model_params,
            model_dir=output_dir,
            config=tf.contrib.learn.RunConfig(save_checkpoints_secs=None, save_checkpoints_steps=check_n_iter),
        )
     
        return tf.contrib.learn.Experiment(
            estimator = estimator,
            train_input_fn = train_input_fn,
            eval_input_fn = eval_input_fn,
            train_steps = num_epochs,
            eval_steps = eval_steps,
            train_monitors = [logging_hook],
            eval_metrics = eval_metric_add,
            export_strategies = [
                saved_model_export_utils.make_export_strategy(
                model.predict_input_fn,
                exports_to_keep=1
            )],
            **experiment_args
        )
    return _experiment_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-file-pattern',
        help='GCS or local paths to training data',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--eval-file-pattern',
        help='GCS or local paths to eval data',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--num-epochs',
        help="""\
        Maximum number of training data epochs on which to train.\
        """,
        type=int,
        default=2000
    )
    parser.add_argument(
        '--batch-size',
        help='Batch size',
        type=int,
        default=128
    )
    parser.add_argument(
        '--shuffle',
        help='Shuffle samples or not',
        type=bool,
        default=True
    )
    parser.add_argument(
        '--eval-steps',
        help='Number of steps to run evalution for at each checkpoint',
        default=10,
        type=int
    )
    parser.add_argument(
        '--check-n-iter',
        help='Number of iterations per check',
        default=100,
        type = int
    )
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )

    parser.add_argument(
        '--tfrecord-compress-type',
        help='tfrecord compress type with gzip, zlib and none',
        default='gzip'
    )

    # Arguments for hyper-parameter tuning
    parser.add_argument(
        '--learning-rate',
        help='the tunable learning rate',
        type=float,
        default=0.0005
    )

    parser.add_argument(
        '--num-filters',
        help='Number of filters in the convolutional layers',
        type=int,
        default=8
    )

    parser.add_argument(
        '--dropout-prob',
        help='dropout layer probability',
        type=float,
        default=0.20
    )

    parser.add_argument(
        '--dense-layer1-size',
        help='dense layer 1 size',
        type=int,
        default=20
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

    # Experiment arguments
    parser.add_argument(
        '--eval-delay-secs',
        help='How long to wait before running first evaluation',
        default=10,
        type=int
    )
    parser.add_argument(
        '--min-eval-frequency',
        help='Minimum number of training steps between evaluations',
        default=100,
        type=int
    )

    args = parser.parse_args()
    arguments = args.__dict__
    tf.logging.set_verbosity(arguments.pop('verbosity'))

    job_dir = arguments.pop('job_dir')

    # Combine all the tunable hyper-parameters into a dict
    model_params = {
                    'learning_rate': arguments.pop('learning_rate'),
                    'num_filters': arguments.pop('num_filters'),
                    'dropout_prob': arguments.pop('dropout_prob'),
                    'dense_layer1_size': arguments.pop('dense_layer1_size')
                   }

    print('model_params : {}'.format(model_params))

    print('Starting Census: Please lauch tensorboard to see results:\n'
            'tensorboard --logdir=$MODEL_DIR')

    # Run the training job
    # learn_runner pulls configuration information from environment
    # variables using tf.learn.RunConfig and uses this configuration
    # to conditionally execute Experiment, or param server code
    learn_runner.run(generate_experiment_fn(model_params = model_params, **arguments), job_dir)
