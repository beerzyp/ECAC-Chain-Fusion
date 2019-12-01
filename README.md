# audio-classifier-keras-cnn
Audio Classifier in Keras using Convolutional Neural Network

*DISCLAIMER: This code is not being maintained.  Your Issues will be ignored.   For up-to-date code, switch over to [Panotti](https://github.com/drscotthawley/panotti).*


O que fazer
    Audio Classification
        Feature Extraction -> CNN used
    Video Classification
        Feature Extraction -> also CNN
    Fusion
        Merging audio and video features


EXAMPLE: https://github.com/drscotthawley/audio-classifier-keras-cnn

# Project Description 

Achieve a gold description using co-learning:

1. http://vsubhashini.github.io/language_fusion.html (Improving LSTM-based Video Description with Linguistic Knowledge Mined from Text); 
2. Taking this problem as a base work, improve by using sound to improve the LSTM model (RNN model); 
3.  Mainly, for application, taking in considerations the nouns of the description, see if it's truth by the sounds that are made to achieve a better gold description



# Videos Tutorial:

[RNN](https://www.youtube.com/watch?v=_h66BW-xNgk&t=657s])





# Main Paper

Improving LSTM-based Video Description with Linguistic Knowledge Mined from Text LINK: https://arxiv.org/abs/1604.01729

## Notes

1. The input ~x to the first LSTM layer is a sequence of frame features obtained from the penultimate layer (fc 7 ) of a Convolutional Neural Network (CNN) after the ReLu operation. This LSTM layer encodes the video sequence.
After viewing all the frames, the second LSTM layer learns to decode this
state into a sequence of words

2. Early Fusion. Our first approach (early fusion), is to pre-train portions of the network modeling language on large corpora of raw NL text and then continue “fine-tuning” the parameters on the paired video-text corpus

3. Late Fusion. Our late fusion approach is similar to how neural machine translation models incorporate a trained language model during decoding. At each step of sentence generation, the video caption model proposes a distribution over the vocabulary.

4. Deep Fusion. In the deep fusion approach (Fig. 2), we integrate the LM a step deeper in the generation process by concatenating the hidden state of the language model LSTM (h LM) with the hidden state t of the S2VT video description model (h V t M ) and use the combined latent vector to predict the output word.

5. _Evaluation Metrics_: We evaluate performance using machine translation (MT) metrics to compare the machine generated descriptions to human ones


## Other Papers related

This section contains a list of related papers.

- [Audio Set: An ontology and human-labeled dataset for audio events](https://ieeexplore.ieee.org/abstract/document/7952261/keywords#keywords)

- [Robust Sound Event Classification Using Deep Neural Networks](https://ieeexplore.ieee.org/document/7003973)

- [Hierarchical classification of audio data for archiving and retrieving](https://ieeexplore.ieee.org/document/757472)

- [Audio Classification Method Based on Machine Learning](https://ieeexplore.ieee.org/document/8047110)

- [AUDIO FEATURE EXTRACTION AND ANALYSIS FORSCENE SEGMENTATION AND CLASSIFICATION](https://static.aminer.org/pdf/PDF/000/290/667/a_video_mosaicking_technique_with_self_scene_segmentation_for_video.pdf)

 - [Convolutional recurrent neural networks for music classification](https://arxiv.org/pdf/1609.04243.pdf)



### Speech Recognition (FESR):

#### Paper:

[Recent Advances in the Automatic Recognition of
Audio-Visual Speech](http://www.ifp.illinois.edu/~ashutosh/papers/IEEE%20AVSR.pdf)

#### Repos:

Biblioteca de pyhon Audioanalysis:

    https://github.com/xiao2mo/pyAudioAnalysis


https://github.com/gionanide/Speech_Signal_Processing_and_Classification (using feature extraction code from this one)

https://github.com/Angeluz-07/audio-processing-files (audio mp3 files with extracted dataset examples)

https://github.com/Angeluz-07/audio-processing-data (results from feature extraction)

https://www.kaggle.com/ashishpatel26/feature-extraction-from-audio  (using for test sets)

https://musicinformationretrieval.com/mfcc.html (Librosa conversion)

## DataSets

1. http://research.google.com/audioset/dataset/index.html

Contains classification of the audio

### Microsoft Video Description Dataset (Youtube videos):

2. Main paper datasets used

[MPII Movie Description (MPII-MD) Dataset](http://www.mpi-inf.mpg.de/movie-description)

[Montreal Video Annotation Description (M-VAD) Dataset](http://www.mila.umontreal.ca/Home/public-datasets/montreal-video-annotation-dataset)

# Concepts

1. RNN: recurrent Neural Network
2. Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture[1] used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (such as images), but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition[2], speech recognition[3][4] and anomaly detection in network traffic or IDS's (intrusion detection systems). A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell. Source: https://en.wikipedia.org/wiki/Long_short-term_memory
3. Machine Translation (MT): sometimes referred to by the abbreviation MT (not to be confused with computer-aided translation, machine-aided human translation (MAHT) or interactive translation) is a sub-field of computational linguistics that investigates the use of software to translate text or speech from one language to another. Source: https://en.wikipedia.org/wiki/Machine_translation
4. Language Models (LMs):A statistical language model is a probability distribution over sequences of words. Given such a sequence, say of length m, it assigns a probability P ( w 1 , … , w m ) {\displaystyle P(w_{1},\ldots ,w_{m})}  to the whole sequence. The language model provides context to distinguish between words and phras that sound similar. For example, in American English, the phrases "recognize speech" and "wreck a nice beach" sound similar, but mean different things.
5. S2VT: uses a sequence to sequence approach that maps an input ~x = (x 1 , ... , x T ) video frame feature sequence to a fixed dimensional vector and then decodes this into a sequence of output words ~y = (y 1 , ... , y N ). Source: Main Paper
Convolutional Neural Network (CNN):






# Presentation ideas

The goals description is determined by crowdsourced human evaluations;

We are going to approach the problem as a classification of audiovisual speech instead of a golden standard problem as we lack a proper dataset for golden standard audiovisual recognition.

In practice, using an ensemble of networks trained slightly differently can improve performance (Early fusion, Late Fusion, Deep Fusion)

We choose Late Fusion approach
 




# How to run
1. Download AudioSet dataset
    * Each folder represents a class
    * Command: `python3 download_audioset.py`
2. Preprocess data
    * ATM not working with AudioSet, but rather with the Samples folder
    Command: `python3 preprocess_data.py`
3. Train Network (Audio only)
    Command: `python3 train_network.py`
    
Future Work:
    * Convert mp4 files from AudioSet to .wav files and use these to train the network
    * Evaluate Results
    * Add images from the video to this analysis (re-train network)
    * Evaluate Results
