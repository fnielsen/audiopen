import audiopen.gendervoice

samples = audiopen.gendervoice.samples_mono_11025(metadata, yieldgender=True)
for sample, gender in samples:
    pitch = np.median([pitch for pitch in audiopen.gendervoice.samples_to_pitches(sample)])
    print("{} {}".format(pitch, gender))
