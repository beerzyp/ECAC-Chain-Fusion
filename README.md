Research in Multimodal Machine Learning has been growing in importance and interest, because it can provide major advantages and results, when compared to unimodal approaches. Methods in this area have reached fields like multimedia classification, audiovisual automatic speech recognition, generation of natural language descriptions of videos, among other applications.

This project introduces a new method to handle multimodal tasks, to which we call Chain Fusion. We provide the main theoretical properties of our method, comparing it to current approaches, such as early and Late Fusion. We also developed an empirical study where we compare unimodal learners, Late Fusion and Chain Fusion. Results show that our method is able to reach performance levels as good or even better than previous methods used in this field.



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
    * Command: `python3 preprocess_data.py`
3. Train Network (Audio only)
    * Command: `python3 train_network.py`
   
