# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:55:16 2023

@author: mrostami21
"""

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# 1. Load the GTZAN dataset
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
data = []
labels = []

for genre in genres:
    print(f"Processing genre: {genre}")
    for idx, filename in enumerate(os.listdir(f'genres/{genre}')):
        print(f"  Processing file {idx+1} of {len(os.listdir(f'genres/{genre}'))}: {filename}")
        try:
            y, sr = librosa.load(f'genres/{genre}/{filename}')
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            data.append(mfcc)
            labels.append(genres.index(genre))
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    print(f"Finished processing genre: {genre}\n")

# Zero-padding the data to make all MFCCs the same shape
max_length = max(mfcc.shape[1] for mfcc in data)  
padded_data = [np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1]))) for mfcc in data]
X = np.array(padded_data)

print("Data loaded and processed.\n")

# 2. Preprocess the data
y = to_categorical(np.array(labels), num_classes=len(genres))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3a. Traditional Models
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print("Training Random Forest Classifier...")
rf = RandomForestClassifier()
rf.fit(X_train_flat, y_train.argmax(axis=1))
rf_preds = rf.predict(X_test_flat)
rf_acc = accuracy_score(y_test.argmax(axis=1), rf_preds)
print("Random Forest trained.\n")

print("Training Support Vector Machine...")
svm = SVC()
svm.fit(X_train_flat, y_train.argmax(axis=1))
svm_preds = svm.predict(X_test_flat)
svm_acc = accuracy_score(y_test.argmax(axis=1), svm_preds)
print("SVM trained.\n")

# 3b. Deep Learning Models
print("Training Simple Neural Network...")
nn_model = Sequential()
nn_model.add(Flatten(input_shape=(X_train.shape[1], X_train.shape[2])))
nn_model.add(Dense(512, activation='relu'))
nn_model.add(Dense(256, activation='relu'))
nn_model.add(Dense(len(genres), activation='softmax'))
nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)
nn_acc = nn_model.evaluate(X_test, y_test)[1]
print("Simple Neural Network trained.\n")

print("Training Convolutional Neural Network...")
X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]
cnn_model = Sequential()
cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(len(genres), activation='softmax'))
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train, epochs=30, batch_size=32, validation_split=0.1)
cnn_acc = cnn_model.evaluate(X_test_cnn, y_test)[1]
print("CNN trained.\n")

# 4. Visualize the results
models = ['Random Forest', 'SVM', 'Simple NN', 'CNN']
accuracies = [rf_acc, svm_acc, nn_acc, cnn_acc]
plt.bar(models, accuracies)
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.show()
