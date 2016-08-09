#!/usr/bin/env python
"""learn_gender.py

Learn gender for the gendervoice dataset.

"""

from __future__ import division, print_function

from keras.models import Sequential
from keras.layers import Dense, Activation

import audiopen.gendervoice


# Keras deep learning
model = Sequential()
model.add(Dense(output_dim=64, input_dim=15))
model.add(Activation("relu"))
model.add(Dense(output_dim=2))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])


# Load metadata, and the data if that is not already downloaded
metadata = audiopen.gendervoice.load_metadata()
audiopen.gendervoice.download_all(metadata)

feature_extractor = audiopen.gendervoice.FeatureExtractor()


# Iterate over all audio files and get pitch.
filenames = audiopen.gendervoice.get_filenames(metadata)

pitches_for_all = []
for filename, name, gender in zip(
        filenames, metadata.personLabel, metadata.genderLabel):
    try:
        samples = audiopen.gendervoice.get_samples_mono_11025(filename)
    except wave.Error:
        continue
    print(filename)
    feature_set = np.array([features
                            for features in audiopen.gendervoice.iter_samples_to_features(samples)])
    Y = np.repeat([[1, 0]] if gender == 'male' else [[0, 1]], feature_set.shape[0], axis=0)
    model.train_on_batch(feature_set, Y)
