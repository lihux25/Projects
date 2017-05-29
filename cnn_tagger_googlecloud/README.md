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
<img src="images/preprocess_pipeline.png" width="250">

## CNN network
<img src="images/cnn_training.png" width="600">

## Hyper-parameter tuning
<img src="images/hptuning.png" width="600">
