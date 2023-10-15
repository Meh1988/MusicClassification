# MusicClassification
Music Genre Classification
Classify music genres using the GTZAN dataset. The project employs a comparison of different models, both from the traditional machine learning domain and deep learning, based on their accuracy.

Table of Contents
Dataset
Prerequisites
Models Used
Results

Dataset
The GTZAN dataset is utilized for this project. It is a collection of 1000 audio tracks, each lasting 30 seconds. The dataset comprises 10 genres, each represented by 100 tracks. The audio files are in .wav format with properties of 22050Hz mono 16-bit.
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification


Prerequisites
Ensure you have the following libraries installed:
librosa
numpy
matplotlib
scikit-learn
keras
ffmpeg-python
Installation


Models Used
Random Forest: Utilizes a collection of decision trees for predictions.
SVM (Support Vector Machine): Finds a hyperplane that best divides the dataset into classes.
Simple Neural Network: A feedforward neural network with several dense layers.
CNN (Convolutional Neural Network): Uses convolutional layers to extract features from MFCCs.

Results
The accuracies of the models are visualized with a bar graph, detailing performance metrics and allowing for a direct comparison between the models.
