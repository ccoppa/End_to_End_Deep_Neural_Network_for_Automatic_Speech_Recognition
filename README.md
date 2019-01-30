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
- CNNs are popular in image analys as they are excellent for image feature extractions. Since audio data can be represented in Spectrogram (or MFCC) and Spectrogram (or MFCC) can be thought of as a visual representation of speech, CNN can be applied on top of RNN to complement each other for better performance. The job of CNN in this model is to convert Spectrogram image data into higher representations or features for the rest of the acoustic model. 
- Gated recurrent unit (GRU), a variant of LSTM, can track time series data through memory. This type of neural networks have temporal memory as it uses a gating mechanism to ensure proper propagation of information through time steps. This temporal memory is an important characteristics for training and decoding speech.
- One shortcoming of conventional RNNs is that they are only able to make use of previous context. In speech recognition, where whole utterances are transcribed at once, there is no reason not to exploit future context as well. Bidirectional RNNs do this by processing the data in both directions with two separate hidden layers which are then fed forwards to the same output layer.
- Since GRUs produce "probability densities" over each time step, we will have the sequencing issue. 
- Sequencing issue is that the number of frames does not have a predictible correspondence to the number of the output symbols eg. phonemes, graphemes, or words. For example, if we speak the same word "speech" in two different speed ie. consider saying "speech" vs "speeeech", the length of input signals are different, but our ASR should decode both as the same six-letter word, "speech".
- The RNN could learn what those graphemes should be if there was a label associated with each frame, but this would require some sort of manual pre-segmentation and is not a practical approach for implementation. 
- A more ideal solution would be to provide the network with a loss function across the entire label sequence that it could minimize when training. We would like the probability distribution of the softmax output to "spike" for each grapheme and provide blanks or some other consistently ignored result between the graphemes so that the transcription could be easily decoded. This would solve the sequencing problem as audio signals of arbitrary length are converted to text.
- Connectionist Temporal Classification (CTC) loss function can be used to train the networks and to help solving the sequencing issue.
- CTC decodes the most likely symbol from each softmax distribution and results in a string of symbols in a length that is equivalent to the original input sequence (the frames). However, with the CTC training, the probable symbols have become consolidated. The CTC decoding algorithm can then compress the transcription to its correct length by ignoring adjacent duplicates and blanks.
- Combining the CTC decoding algorithm with the softmax output layer would convert the RNN outputs into text. The sequence-to-sequence transformation of our ASR is then achieved! 
- [Not implemented in this project]: A more complex CTC decoding can provide not only the most likely transcription, but also some number of top choices using a beam search of arbitrary size. This is useful if the result will then be processed with a language model for additional accuracy. 


## Model training
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


