#!/usr/bin/env python
"""gender_pitches.py

Shows median pitch for the files in the gendervoice dataset.

"""

import numpy as np

import audiopen.gendervoice


metadata = audiopen.gendervoice.query_metadata()
# audiopen.gendervoice.download_all(metadata)

samples = audiopen.gendervoice.iter_samples_mono_11025(
    metadata, yieldgender=True)
for sample, gender in samples:
    pitches = audiopen.gendervoice.iter_samples_to_pitches(sample)
    pitch = np.median([pitch for pitch in pitches])
    print("{} {}".format(pitch, gender))
