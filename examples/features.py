

from audiopen.gendervoice import (
    load_metadata, get_filenames, get_samples_mono_11025,
    iter_samples_to_features)

metadata = load_metadata()
filename = get_filenames(metadata)[0]
samples = get_samples_mono_11025(filename)

for features in iter_samples_to_features(samples):
    print(features)



