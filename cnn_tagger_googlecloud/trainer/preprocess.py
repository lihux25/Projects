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

from __future__ import absolute_import

import argparse
import csv
import datetime
import errno
import io
import logging
import subprocess
import sys
import re

from StringIO import StringIO

import apache_beam as beam
from apache_beam.metrics import Metrics
from apache_beam.utils.pipeline_options import PipelineOptions
from apache_beam.utils.pipeline_options import SetupOptions

from PIL import Image
import tensorflow as tf
import numpy as np

from tensorflow.contrib.slim.python.slim.nets import inception_v3 as inception
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io

class ReadFileAndConvert(beam.DoFn):
    """Read the npy file and convert them to numpy arrays with some filter requirement."""

    def __init__(self):
        super(ReadFileAndConvert, self).__init__()

    def process(self, element):
        """
        Args:
        element: the element being processed

        Returns:
        The processed element.
        """

        fields = element.split(',')
        filepath = fields[0]
        if filepath.startswith('gs://'):
            try:
                f = StringIO(file_io.read_file_to_string(filepath))
                arr = np.load(f)
            except ValueError:      
                print('bad file: {:s}'.format(fields[0]))
        else:
            try:
                arr = np.load(fields[0])
            except ValueError:
                print('bad file: {:s}'.format(fields[0]))
    
        evt_arr = arr['event']
        jet_idx_arr = arr['ak7pfcand_ijet']

        list_mapper = []
        for evt, jet_idx, arr_entry in zip(evt_arr, jet_idx_arr, arr):
            if arr_entry['jet_pt_ak7'] < 100:
                continue
            if ('TT' in fields[1] and arr_entry['jet_isW_ak7'] !=1) or ('QCD' in fields[1] and arr_entry['jet_isW_ak7'] !=0):
                continue
            list_mapper.append( ((evt, jet_idx, fields[1]), arr_entry) )
        return list_mapper

class ProcessImage(beam.DoFn):

    def __init__(self, image_dims=(30, 30), channels=3):
        super(ProcessImage, self).__init__()
        self.image_dims = image_dims
        self.channels = channels

    def process(self, element):
        """

        """
        key = element[0]
        val = np.array(list(element[1]))

    # Function to fill the object images
        def _fill_jet_images(pfcands):
            nx = self.image_dims[0]
            ny = self.image_dims[1]
            xbins = np.linspace(-1.4,1.4,nx+1)
            ybins = np.linspace(-1.4,1.4,ny+1)
        # Treat the charge of the particle candidate as an additional channel (analogy to the rgb channel in computer vision)
            if self.channels == 3:
            # Charges are within values of -1, 0 or 1
                charge_dims = (-1, 0, 1)
            else:
                charge_dims = (-99) 
            this_jet_images = np.zeros((nx, ny, self.channels))
            # Process the rgb channels
            for ic in range(self.channels):
                target_charge = charge_dims[ic]
                # Filter out rows with only target_charge. If it's -99, then no filtering.    
                if target_charge != -99:
                    pfcands_ij = pfcands[pfcands['ak7pfcand_charge']==target_charge]
                else:
                    pfcands_ij = pfcands
                if( len(pfcands_ij) == 0 ):
                    continue
                # Normalize the jet image to be the first candidate coordiates
                x = pfcands_ij['ak7pfcand_eta'] - pfcands_ij['ak7pfcand_eta'][0]
                y = pfcands_ij['ak7pfcand_phi'] - pfcands_ij['ak7pfcand_phi'][0]
                # The weights (analogy to color channel intensity) are the transverse momentum of the candidates
                weights = pfcands_ij['ak7pfcand_pt']
                # Processing to compose a jet "image"
                hist, xedges, yedges = np.histogram2d(x, y, weights=weights, bins=(xbins, ybins))
                for ix in range(nx):
                    for iy in range(ny):
                        # Fill the image to a numpy array
                        this_jet_images[ix, iy, ic] = hist[ix, iy]
            return this_jet_images
         
        this_jet_images = _fill_jet_images(val)
#        this_jet_images = this_jet_images[this_jet_images!=0] # for testing since so many 0's
    
        return [(key, this_jet_images)]

class TFExampleFromImage(beam.DoFn):
    """
    Convert the jet images (in numpy arrays) to TFExample format
    """

    def __init__(self):
        self.tf_session = None
        self.graph = None

    def start_bundle(self, context=None):
        # There is one tensorflow session per instance of TFExampleFromImage.
        # The same instance of session is re-used between bundles.
        # Session is closed by the destructor of Session object, which is called
        # when instance of TFExampleFromImage() is destructed.
        if not self.graph:
            self.graph = tf.Graph()
            self.tf_session = tf.InteractiveSession(graph=self.graph)

    def process(self, element):

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        try:
            element = element.element
        except AttributeError:
            pass

        key, this_jet_image = element

        image_raw = this_jet_image.tostring()

        height = this_jet_image.shape[0]
        width = this_jet_image.shape[1]
        depth = this_jet_image.shape[2]

        proc_type = str(key[2])
        if 'TT' in proc_type:
            label = 1
        elif 'QCD' in proc_type:
            label = 0
        else:
            label = -1

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'proc_type': _bytes_feature(proc_type),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_raw),
        }))

        yield example

def run(argv=None):
    """Main entry point; defines and runs the pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                          dest='input',
                          default='filenames.csv',
                          help='Input file to process.')
    parser.add_argument('--output',
                          dest='output',
                          default='output_from_beam',
                          help='Key string of the output file name to write results to.')
    parser.add_argument('--output_path',
                          default='preprocess_output',
                          help='Path directory to put the results to.')

    known_args, pipeline_args = parser.parse_known_args(argv)

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    p = beam.Pipeline(options=pipeline_options)

    # Preparing the images. At this stage, we have images in shape of (HEIGHT, WIDTH, CHANNELS)
    prepimages = (p
                  | 'read' >> beam.io.ReadFromText(known_args.input) 
                  | 'parse npy file and filtered' >> beam.ParDo(ReadFileAndConvert())
                  | 'group' >> beam.GroupByKey()
                  | 'process data into image' >> beam.ParDo(ProcessImage()))

    # Write out the jet images into text files for easy checking and validation
    _ = (prepimages 
            | 'remove zeros' >> beam.Map(lambda (key, val): (key, val[val!=0]))
            | 'write' >> beam.io.WriteToText(known_args.output_path + '/' + known_args.output, file_name_suffix='.txt.gz') )

    # Convert the jet images into the tfrecords and compress them as well
    _ = (prepimages 
            | 'make TFExample' >> beam.ParDo(TFExampleFromImage())
            | 'SerializeToString' >> beam.Map(lambda x: x.SerializeToString())
            | 'Save to disk'
               >> beam.io.WriteToTFRecord(known_args.output_path + '/' + known_args.output,
                                  file_name_suffix='.tfrecords.gz'))

  # Actually run the pipeline (all operations above are deferred).
    result = p.run()
    result.wait_until_finish()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
