"""Handle data.

Examples
--------
>>> import audiopen.data
>>> audio_segments = audiopen.data.audio_segments()
>>> audio_segment = audio_segments.next()
>>> import pydub.playback
>>> # pydub.playback.play(audio_segment)

>>> audio_segments = audiopen.data.audio_segments_mono_11025()
>>> audio_segment = audio_segments.next()
>>> audio_segment.channels
1

"""

import errno

from os import makedirs
from os.path import exists, expanduser, join, split

from urllib2 import urlopen

import numpy as np

from pydub import AudioSegment


from .metadata import read_metadata


DATA_DIRECTORY = join(expanduser('~'), 'data', 'audiopen')


def download(directory=DATA_DIRECTORY):
    """Download data from Wikimedia Commons.

    This will download all audio files mentioned in the metadata file, 
    if they are not already downloaded.

    Parameters
    ----------
    directory : str
        Local directory where the audio files are to be stored.

    Examples
    --------
    >>> download()

    """
    # Make directory if not exist
    try:
        makedirs(directory)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    metadata = read_metadata()
    
    # Download audio files
    for remote_filename in metadata.Link:
        _, filename = split(remote_filename)
        local_filename = join(directory, filename)
        if not exists(local_filename):
            # Buffered output
            # https://gist.github.com/hughdbrown/c145b8385a2afa6570e2
            buffer = urlopen(remote_filename)
            with open(local_filename, 'wb') as output:
                while True:
                    data = buffer.read(4096)
                    if data:
                        output.write(data)
                    else:
                        break


def filenames(directory=DATA_DIRECTORY, yieldcategory=False):
    """Yield filenames for audio files.

    Parameters
    ----------
    directory : str
        Local directory where the audio files are to be stored.

    Yields
    ------
    filename : str
        Local filename for audio file

    """
    metadata = read_metadata()
    for index, row in metadata.iterrows():
        remote_filename = row.Link
        _, filename = split(remote_filename)
        local_filename = join(directory, filename)
        if yieldcategory:
            yield local_filename, row.Category
        else:
            yield local_filename


def audio_segments(directory=DATA_DIRECTORY, yieldcategory=False):
    """Yield audio segments objects.

    Yields
    ------
    audio_segment : pydub.AudioSegment
        Audio segments.

    """
    for filename, category in filenames(directory=directory,
                                        yieldcategory=True):
        audio_segment = AudioSegment.from_file(filename)
        if yieldcategory:
            yield audio_segment, category
        else: 
            yield audio_segment


def audio_segments_mono_8k(directory=DATA_DIRECTORY):
    """Yield audio segments in mono and 8 kHz.

    Yields
    ------
    audio_segment : pydub.AudioSegment
        Resampled audio segments in mono.

    """
    for audio_segment in audio_segments(directory=directory):
        mono_8k = audio_segment.split_to_mono()[0].set_frame_rate(8000)
        yield mono_8k


def audio_segments_mono_11025(directory=DATA_DIRECTORY, yieldcategory=False):
    """Yield audio segments in mono and 11025 Hz.

    The first channel is returned after a split to mono.

    Yields
    ------
    audio_segment : pydub.AudioSegment
        Resampled audio segments in mono.

    Examples
    --------
    >>> audio_segments = audio_segments_mono_11025()
    >>> audio_segment = audio_segments.next()
    >>> audio_segment.channels
    1
    >>> audio_segment.frame_rate
    11025

    """
    for audio_segment, category in audio_segments(directory=directory,
                                                  yieldcategory=True):
        output = audio_segment.split_to_mono()[0].set_frame_rate(11025)
        if yieldcategory:
            yield output, category
        else:
            yield output


def samples_mono_11025(directory=DATA_DIRECTORY, yieldcategory=False):
    """Yield audio segments in mono and 11025 Hz.

    The first channel is returned after a split to mono.

    Yields
    ------
    samples : numpy.array
        Resampled audio segments in mono as numpy array
    category : str
        Category of sample represented as a string. It is only yielded if 
        yieldcategory is True

    Examples
    --------
    >>> samples = samples_mono_11025()
    >>> sample = samples.next()

    """
    for audio_segment, category in audio_segments_mono_11025(
            directory=directory, yieldcategory=True):
        output = np.array(audio_segment.get_array_of_samples())
        if yieldcategory:
            yield output, category
        else:
            yield output
