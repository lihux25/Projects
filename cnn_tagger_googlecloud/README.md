# Predicting a W Boson Particle Using a Convolutional Neural Network and Google Cloud Platform

(WORK IN PROGRESS for the README file...)

## The data and my goal
 * What do we get from the CMS detector?
	 * The [CMS detector](https://cms.cern/detector) like a giant digital camera taking pictures of particles passing through it. When interacting with the detector materials, the particles deposit energy and expose their positions which are recorded, digitizied and stored. Some algorithms are used to connect the digital signals in different detector parts to identify some very basic types of stable particles, e.g., electron and pion. However more advanced methods are needed to identify heavier particles that decay to multiple of the stable particles. **This is the primary goal of the work done here, i.e., to identify a heavy W boson particle** (similarly can be extended to other particles).
 * What are in the data and some of its features?
 	* The data I use come from the public dataset from the CMS collaboration. They are simulated events with detailed detector configuration and setup. Therefore we know the answer whether the bunch of particles we look at come from the decay of a heavier particle.
 	* Given enough boost of a particle, all its decay product will concentrate in a confined region within a solid angle. Here I only consider the 2D projected sphere as our "picture" which is an analogy to our printed pictures in books and magazines. The size convention does not matter but we have a keyword "ak7" for all the related qualities.
 	* The pictures of the signal (targeted W boson particle) and other noise can be viewd in this [jupyter notebook](cnn_tagger.ipynb) (they are averaged over many of them). Here I use the particle energy as a measure of the intensity level (for a cell/"pixel") and particle charges as the channels (analogy to the RGB channels).
 * Why convolutional neural network?
 	* Given the similarity between the CMS detector data and our everyday photos, it is a good starting point to consider the convolutional neural network which works very well for imagine recognition.
 
## General project design consideration
 * Given I have roughly 5 million original data points to process, I use Google Cloud Dataflow service for data pre-processing pipeline for distributed computing.
 * I also need distributed training and some hyper-parameter tuning.
 * I use high level tensorflow and Keras API to quickly build the network and training pipeline.
 
## Data pre-processing pipeline
 * The data pre-processing is done using Google Cloud Dataflow service with the Beam framework. The source files (in npy format) are copied to a Google Cloud Storage bucket gs://jetimage-source-files/ and divided into train and test parts.
 * Two csv files are used to compile all the file location in Google Cloud Storage bucket and some label/notation information:
	* gs://jetimage-source-files/filenames\_gcloud\_train.csv
	* gs://jetimage-source-files/filenames\_gcloud\_test.csv
 * The output files in compressed tfrecords format are in gs://jetimage-tfrecords bucket
 * The pipeline is defined in the [preprocess.py](trainer/preprocess.py) file in Beam framework. The major parts in the pipeline is to apply selection cuts and assign correct labels (the ReadFileAndConvert function), convert the raw information in particles to an image (the ProcessImage function), and convert the image (in numpy array) to tfrecords (the TFExampleFromImage function) so that they can be used by the ML Engine as input. The flow chart can be viewed as follows:
 
<img src="images/preprocess_pipeline.png" width="250">

## Covnet definition and training
 * The definition of the model is [model.py](trainer/model.py). Function build_conv_model uses Keras API to define the whole Covnet structure with 3 tuning parameters. The build_read_and_decode_fn provides input batchs with images and labels for both training and testing. The model_fn define the loss, optimization method and evaluation metrics. 
 * In the [task.py](trainer/task.py) file, build a tf.contrib.learn.Experiment to handle the training and evaluation loops for distributed training.
 * The training results are stored in gs://cnn-tagger/* including the exported model.
 * The structure of the Covnet and training flow chart is extracted from the tensorboard monitor below.
 
<img src="images/cnn_training.png" width="600">

## Hyper-parameter tuning
 * For a simple case, I only tune the learning_rate which is one of the most important hyper-parameters to tune. The definition of the tuning parameters are in the [hptuning_config.yaml](hptuning_config.yaml) file. 
 * A scan of 5 different learning_rate is shown below. The default value chosen was 1e-4, however a better one is found to be 0.0005 (yellow line) which provide an improved accuracy!
<img src="images/hptuning.png" width="600">
