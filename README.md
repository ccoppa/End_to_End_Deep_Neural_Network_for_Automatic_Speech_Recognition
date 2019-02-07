# An End-to-End DNN for ASR with CTC
[![GPL-3.0 Licensed](https://img.shields.io/badge/License-GPL3.0-blue.svg?style=flat)](https://opensource.org/licenses/GPL-3.0) [![TensorFlow Version](https://img.shields.io/badge/Tensorflow-1.4+-blue.svg)](https://www.tensorflow.org/) [![Keras Version](https://img.shields.io/badge/Keras-2.0+-blue.svg)](https://keras.io/) [![Python Version](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/) 

## Introduction
In this notebook, we will build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline in Keras framework.

## Data Sets 
[LibriSpeech ASR corpus](http://www.openslr.org/12) contains 1000 hours corpus of read English speech. We will be using a subset of the data to train and validate the models.

## Modeling step
- We begin by investigating the LibriSpeech dataset that will be used to train and evaluate the models. 
- Algorithm will first convert the raw audio data to feature representations ([Spectrograms](https://www.youtube.com/watch?v=_FatxGN3vAM) and [MFCCs](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)) that are commonly used for ASR. 
- Build and compare the performance of various neural networks that map the audio features to transcribed text. 
- The main model components are - CNN, GRU and CTC. 
- CNNs are popular in image analys as they are excellent for image feature extractions. Since audio data can be represented in Spectrogram (or MFCC) and Spectrogram (or MFCC) can be thought of as a visual representation of speech, CNN can be applied on top of GRU to complement each other for better performance. The job of CNN in this model is to convert Spectrogram image data into higher representations or features for the rest of the acoustic model. 
- Gated recurrent unit (GRU), a variant of LSTM, can track time series data through memory. This type of neural networks have temporal memory as it uses a gating mechanism to ensure proper propagation of information through time steps. This temporal memory is an important characteristics for training and decoding speech.
- [Variational inference based dropout technique](http://arxiv.org/abs/1512.05287) is applied in our GRU models to alleviate overfitting.
- [Time distributed](https://keras.io/layers/wrappers/) dense layer is applied to GRU outputs to keep one-to-one relations on input and output, so that the GRU output doesn't need to get flatten to randomly interact between different timesteps. This helps finding patterns in temporal data easier. 
- One shortcoming of conventional RNNs is that they are only able to make use of previous context. [Deep bidirectional GRUs](https://www.cs.toronto.edu/~graves/asru_2013.pdf) are implemented to process the data in both directions with two separate hidden layers which are then fed forwards to the same output layer.
- [Batch normalization](https://arxiv.org/pdf/1510.01378.pdf), which uses mini-batch statistics to standardize features, is leveraged to expedite convergence of model training and is applied to both CNN and GRU outputs.
- Since GRUs produce "probability densities" over each time step, we will have the sequencing issue. 
- Sequencing issue is that the number of frames does not have a predictible correspondence to the number of the output symbols eg. phonemes, graphemes, or words. For example, if we speak the same word "speech" in two different speed ie. consider saying "speech" vs "speeeech", the length of input signals are different, but our ASR should decode both as the same six-letter word, "speech".
- The GRU could learn what those graphemes should be if there was a label associated with each frame, but this would require some sort of manual pre-segmentation and is not a practical approach for implementation. 
- A more ideal solution would be to provide the network with a loss function across the entire label sequence that it could minimize when training. We would like the probability distribution of the softmax output to "spike" for each grapheme and provide blanks or some other consistently ignored result between the graphemes so that the transcription could be easily decoded. This would solve the sequencing problem as audio signals of arbitrary length are converted to text.
- [Connectionist Temporal Classification (CTC) loss function](http://www.cs.toronto.edu/~graves/icml_2006.pdf) can be used to train the networks and to help solving the sequencing issue.
- CTC decodes the most likely symbol from each softmax distribution and results in a string of symbols in a length that is equivalent to the original input sequence (the frames). However, with the CTC training, the probable symbols have become consolidated. The CTC decoding algorithm can then compress the transcription to its correct length by collapsing repeated characters not separated by blank.
- Combining the CTC decoding algorithm with the softmax output layer would convert the GRU outputs into text. The sequence-to-sequence transformation of our ASR is then achieved! 
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

## Next steps for further improvement
Besides using more data to train the model, there are two methods that we can try to improve model performance.
1. One standard way to improve the results of the decoder is to incorporate a `language model (LM)`. The job of a LM is to inject the language knowledge into the words-to-text step in ASR, providing another layer of processing between words and text to solve ambiguities in spelling and context. Google AI Language recently published [BERT]( https://arxiv.org/abs/1810.04805) which applies the bidirectional training of `Transformer`, an attention model, to language modelling. BERT seems like a promising approach to try.

2. Alternately, we can try two other popular ASR architectures:
    - attention-based `Seq2Seq` models which powered [Listend-Attend-Spell](https://arxiv.org/abs/1508.01211)
    - `RNN-Transducer`, which can be thought of as an encoder-decoder model, assumes the alignment between input and output tokens is local and monotonic. RNN-Transducer implementation has shown success without LMs.
    
Final words about LM: [Deep Speech 3](http://research.baidu.com/Blog/index-view?id=90)
>Language models are vital to speech recognition because language models can be trained rapidly on much larger datasets, and secondly language models can be used for specializing the speech recognition model according to context (user, geography, application etc.) without a labeled speech corpus for each context. The latter is especially vital in a production speech recognition system.
