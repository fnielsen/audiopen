#!/usr/bin/env python
"""learn_gender.py

Learn gender for the gendervoice dataset.

This script is work in progress. It does not perform well.

"""

from __future__ import division, print_function

import wave

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

import audiopen.gendervoice


# Load metadata, and the data if that is not already downloaded
metadata = audiopen.gendervoice.load_metadata()
audiopen.gendervoice.download_all(metadata)

feature_extractor = audiopen.gendervoice.FeatureExtractor()


# Iterate over all audio files
filenames = audiopen.gendervoice.get_filenames(metadata)


# Keras deep learning
model = Sequential()
model.add(Dense(output_dim=2, input_dim=15))
model.add(Activation("tanh"))
model.add(Dense(output_dim=2))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])


for filename, name, gender in zip(
        filenames, metadata.personLabel, metadata.genderLabel):
    try:
        samples = audiopen.gendervoice.get_samples_mono_22050(filename)
    except wave.Error:
        continue
    feature_set = np.array(
        [features
         for features in audiopen.gendervoice.iter_samples_to_features(samples, sample_rate=22050)])
    proba = model.predict_proba(feature_set, batch_size=32, verbose=0)
    probability_of_female = np.mean(proba, axis=0)[0]
    Y = np.repeat([[1, 0]] if gender == 'male' else [[0, 1]], feature_set.shape[0], axis=0)
    print("Training on {} with gender {} with sample of size {}. Female estimate: {}".format(
        name, gender, Y.shape[0], probability_of_female))
    _ = model.train_on_batch(feature_set, Y)
