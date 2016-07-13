#!/usr/bin/env python
"""gender_pitches.py

Shows median pitch for the files in the gendervoice dataset.

"""

import wave

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib.pyplot import *

import audiopen.gendervoice

# Load metadata, and the data if that is not already downloaded
metadata = audiopen.gendervoice.load_metadata()
audiopen.gendervoice.download_all(metadata)

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
    pitches_for_person = audiopen.gendervoice.get_pitches(samples)
    pitches_for_all.append((pitches_for_person, gender, name, filename))


# Dataframe representation
pitches = pd.DataFrame(pitches_for_all,
                       columns=['pitches', 'gender', 'name', 'filename'])


# Various pitch aggregation functions
def median(pitches):
    """Return median of pitches."""
    return np.median(pitches[:, 0])


def confidence_median(pitches):
    """Return median pitch from pitches with top confience."""
    confidence = pitches[:, 1]
    indices = np.argsort(confidence)
    return np.median(pitches[indices[-min(10, len(indices)):], 0])


# Apply the pitch aggregation functions
pitches['median_pitch'] = pitches.pitches.apply(median)
pitches['confidence_pitch'] = pitches.pitches.apply(confidence_median)


# Stripplot of male vs. female pitches
figure()
sns.stripplot(x=pitches.gender, y=pitches.confidence_pitch, jitter=.1)
ylim(50, 300)
show()
