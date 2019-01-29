# An End-to-End DNN for ASR with CTC
[![GPL-3.0 Licensed](https://img.shields.io/badge/License-GPL3.0-blue.svg?style=flat)](https://opensource.org/licenses/GPL-3.0) [![TensorFlow Version](https://img.shields.io/badge/Tensorflow-1.4+-blue.svg)](https://www.tensorflow.org/) [![Keras Version](https://img.shields.io/badge/Keras-2.0+-blue.svg)](https://keras.io/) [![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/) 

## Introduction
This project uses Keras on deep convolutional neural network, gated recurrent unit and CTC to implement. 
In this notebook, you will build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline!

We begin by investigating the LibriSpeech dataset that will be used to train and evaluate your models. Your algorithm will first convert any raw audio to feature representations that are commonly used for ASR. You will then move on to building neural networks that can map these audio features to transcribed text. After learning about the basic types of layers that are often used for deep learning-based approaches to ASR, you will engage in your own investigations by creating and testing your own state-of-the-art models. Throughout the notebook, we provide recommended research papers for additional reading and links to GitHub repositories with interesting implementations.

* **To run **
Simply run `asr_notebook.ipynb` and results will be automatically saved in the `results` folder.

## Model

### Speech Model

CNN + GRU + CTC

The maximum length of the input audio is 16 seconds, and the output is the corresponding Chinese pinyin sequence. 

* Questions about downloading trained models

The complete source program can be obtained from the archives of the various versions of the software released in the [releases](https://github.com/nl8590687/ASRT_SpeechRecognition/releases) page of Github.

### Language Model 

Maximum Entropy Hidden Markov Model Based on Probability Graph. 

The input is a Chinese pinyin sequence, and the output is the corresponding Chinese character text. 

## About Accuracy

At present, the best model can basically reach 80% of Pinyin correct rate on the test set. 

However, as the current international and domestic teams can achieve 98%, the accuracy rate still needs to be further improved. 

## Python libraries that need importing

* python_speech_features
* TensorFlow
* Keras
* Numpy
* wave
* matplotlib
* math
* Scipy
* h5py
* http
* urllib

## Data Sets 
* **Tsinghua University THCHS30 Chinese voice data set**

  data_thchs30.tgz 
[Download](<http://www.openslr.org/resources/18/data_thchs30.tgz>)

  test-noise.tgz 
[Download](<http://www.openslr.org/resources/18/test-noise.tgz>)

  resource.tgz 
[Download](<http://www.openslr.org/resources/18/resource.tgz>)

* **Free ST Chinese Mandarin Corpus**

  ST-CMDS-20170001_1-OS.tar.gz 
[Download](<http://www.openslr.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz>)

* **AIShell-1 Open Source Dataset** 

  data_aishell.tgz
[Download](<http://www.openslr.org/resources/33/data_aishell.tgz>)

Note：unzip this dataset

```
$ tar xzf data_aishell.tgz
$ cd data_aishell/wav
$ for tar in *.tar.gz;  do tar xvf $tar; done
```

* **Primewords Chinese Corpus Set 1** 

  primewords_md_2018_set1.tar.gz
[Download](<http://www.openslr.org/resources/47/primewords_md_2018_set1.tar.gz>)

Special thanks! Thanks to the predecessors' public voice data set. 

If the provided dataset link cannot be opened and downloaded, click this link [OpenSLR](http://www.openslr.org)

## Logs

Links: [Progress Logs](https://github.com/nl8590687/ASRT_SpeechRecognition/blob/master/log.md)

## Contributors
@ZJUGuoShuai @williamchenwl

@nl8590687 (repo owner)

[**Donate**](https://github.com/nl8590687/ASRT_SpeechRecognition/wiki/donate)

