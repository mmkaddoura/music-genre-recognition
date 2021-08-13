# Music Genre Recognition using CNN and ResNet architectures

## Introduction

Building deep learning models trained on audio data has become an extremely important and lucrative venture over the last decade. Interactive virtual assistants, such as Siri and Alexa, as well as self-driving cars use deep learning models trained on audio data to predict the correct course of action. Companies such as Spotify and Pandora have also based their business around such models, especially on the task of MIR.

MIR (Music Information Retrieval) deals with the analysis of musical content by combining aspects from signal processing, machine learning, and music theory. And MGR (Music Genre Recognition), a subfield of MIR, is significant to companies such as Spotify and Pandora as it enables systems to perform content based music recommendations as well as help organize musical databases. This increase in the importance of MGR deep learning models incentivised us to create this project to analyze audio data to see if we could make accurate MGR predictions.

## Deep Learning Goal

Our overall Deep Learning task is to try come up with a deep learning model that performs better on the GTZAN dataset than the majority of other entries.

We are interested in implementing the framework proposed in the blog ["Music Genre Recognition using Convolutional Neural Networks”](https://towardsdatascience.com/music-genre-recognition-using-convolutional-neural-networks-cnn-part-1-212c6b93da76) by Kunal Vaidya and see if we can recreate the same high accuracy with a CNN or other models.

## Data

For this project, the dataset that we will be working with is the [GTZAN](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification) Genre Classification dataset. This dataset consists of 1,000 audio tracks that are each 30 seconds in length. The songs are split into 10 seperate genres, each represented by 100 tracks:

  + Rock
  + Reggae
  + Pop
  + Metal
  + Jazz
  + Hip-hop
  + Disco
  + Country
  + Classical
  + Blues

## Techniques

For our analysis we were interested in translating the Convolutional Neural Network model from the blog (linked [here](https://towardsdatascience.com/music-genre-recognition-using-convolutional-neural-networks-cnn-part-1-212c6b93da76)) from tensorflow to pytorch and see if we can replicate or even improve upon the accuracy.

We first performed some preprocessing of the dataset by only using the first 3 seconds of each audio file and converting them from a Waveform Audio File Format to a mel-spectrogram. We converted them as mel spectrograms will give us a visual representation of spectrum of frequencies over time. We used the librosa library to perform this conversion. Below are some examples of mel spectrograms for different genres of songs. Please take note that as the mel spectrograms have some observable differences, it gives us confidence that a deep learning model can correctly classify the songs.



We the created a custom CNN model based off the blog's CNN design through tensorflow, as well as a ResNet18 model which we then fine tuned. Based on our analysis, we found the following techniques to be helpful in achieving a high validation accuracy:

  1. **Learning Rate Scheduling:** Instead of using a fixed learning rate, we used a learning rate scheduler, which changed the learning rate after every batch of training. We used the One Cycle Learning Rate Policy.

  2. **Weight Decay:** We added weight decay to the optimizer, a regularization technique which prevents the weights from becoming too large by adding an additional term to the loss function.

  3. **Gradient Clipping:** We also added gradient clipping, which helps limit the values of gradients to a small range to prevent undesirable changes in model parameters due to large gradient values during training.

## Results

The following are our results for the CNN and the ResNet18 models for their last epoch:

|  Model  |  Validation Loss | Validation Accuracy |
|:-------:|:----------------:|:-------------------:|
| CNN     | 0.74             | 0.74                |
| ResNet  | 0.20             | 0.94                |

While we did not achieve the same accuracy with our CNN model as the CNN model from the [blog](https://towardsdatascience.com/music-genre-recognition-using-convolutional-neural-networks-cnn-part-1-212c6b93da76), our ResNet18 model performed extremely well

## Next Steps

As we were not able to include the entire dataset from GTZAN due to time constraints, we would love to go back and finish converting the rest of the WAV files to mel spectrograms. We would also like to edit our CNN model to see if we could imporve its accuracy as well.

