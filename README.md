# An End-to-End DNN for ASR with CTC
[![GPL-3.0 Licensed](https://img.shields.io/badge/License-GPL3.0-blue.svg?style=flat)](https://opensource.org/licenses/GPL-3.0) [![TensorFlow Version](https://img.shields.io/badge/Tensorflow-1.4+-blue.svg)](https://www.tensorflow.org/) [![Keras Version](https://img.shields.io/badge/Keras-2.0+-blue.svg)](https://keras.io/) [![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/) 

## Introduction
In this notebook, we will build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline in Keras framework.

## Data Sets 
[LibriSpeech ASR corpus](http://www.openslr.org/12) contains 1000 hours corpus of read English speech. We will be using a subset of the data to train and validate the models.

## Modeling step
- We begin by investigating the LibriSpeech dataset that will be used to train and evaluate the models. 
- Algorithm will first convert the raw audio data to feature representations (Spectrograms and MFCCs) that are commonly used for ASR. 
- Build and compare the performance of various neural networks that map the audio features to transcribed text. 
- The main model components are - CNN, GRU and CTC. 
- CNNs are popular in image analys as they are excellent for image feature extractions. Since audio data can be represented in spectrogram or MFCC and spectrogram and MFCC can be thought of as an image; therefore CNN can be applied on top of RNN for better performance. 

* **To run **
Simply run `asr_notebook.ipynb` and results will be automatically saved in the `results` folder.

## Python libraries used

* python_speech_features
* TensorFlow
* Keras
* Numpy
* time
* glob
* seaborn
* json
* matplotlib
* librosa
* Scipy
* random
* os
* soundfile
* signal
* contextlib
* requests
* \_pickle


