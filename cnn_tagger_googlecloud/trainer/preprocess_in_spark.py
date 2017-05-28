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
This is almost the same as preprocess.py but in Spark framework. It's not
used because currently it's not straightforward to write out the tfrecords
to serve inputs to tensorflow as in the Beam framework. 
"""

from pyspark import SparkConf, SparkContext
import numpy as np
import tensorflow as tf

conf = SparkConf().setMaster("local[*]").setAppName("Test")
sc = SparkContext(conf = conf)

def ReadFileAndConvert(line):

    fields = line.split(',')
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

# Function to fill the object images
def fill_jet_images(pfcands, image_dims, channels):
    nx = image_dims[0]
    ny = image_dims[1]
    xbins = np.linspace(-1.4,1.4,nx+1)
    ybins = np.linspace(-1.4,1.4,ny+1)
    # Treat the charge of the particle candidate as an additional channel (analogy to the rgb channel in computer vision)
    if channels == 3:
    # Charges are within values of -1, 0 or 1
        charge_dims = (-1, 0, 1)
    else:
        charge_dims = (-99)
    this_jet_images = np.zeros((nx, ny, channels))
    # Process the rgb channels
    for ic in range(channels):
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

def ProcessImage( element ):
    key = element[0]
    val = np.array(list(element[1]))

    this_jet_images = fill_jet_images(val, (30, 30), 3)

    return (key, this_jet_images)

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def TFExampleFromImage( element ):

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
        'height': int64_feature(height),
        'width': int64_feature(width),
        'depth': int64_feature(depth),
        'proc_type': bytes_feature(proc_type),
        'label': int64_feature(label),
        'image_raw': bytes_feature(image_raw),
    }))

    return example

# This is for a local test
input_csv = sc.textFile("./full_filenames.csv")
mappedAndFiltered = input_csv.flatMap(ReadFileAndConvert)
grouped = mappedAndFiltered.groupByKey()

processed = grouped.map(ProcessImage).persist()

toTFExample = processed.map(TFExampleFromImage)
serialize = toTFExample.map(lambda x: x.SerializeToString())

# print or save meaningful (nonzero) entries for viewing/debuging purpose
# so that I can compare with results from jupyter notebook for some examples
# to see that the implementation is correct.
reduceZeros = processed.map(lambda (key, val): (key, val[val!=0]))
results = reduceZeros.collect()

# Unfortunately, there is no easy way of saving tfrecords using spark...
# This code is incomplete only for the final saving...
results2 = serialize.take(10)

print(len(results2))
print(len(results))

for i, result in enumerate(results):
    if i > 10: break
    print(result)
