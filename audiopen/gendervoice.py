"""Gendervoice.

Examples
--------
>>> import audiopen.gendervoice
>>> metadata = audiopen.gendervoice.load_metadata()

"""

import errno

from hashlib import sha256

from os import makedirs
from os.path import exists, expanduser, join, splitext

from string import split

from urllib2 import urlopen

from urlparse import urlsplit

import wave

import aubio

import numpy as np

from pandas import DataFrame, read_csv

from pydub import AudioSegment

from pysoundcard import Stream

import sparql


DATA_DIRECTORY = join(expanduser('~'), 'data', 'audiopen')

METADATA_FILENAME = 'gendervoice.csv'

QUERY_METADATA = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>

SELECT ?person ?personLabel ?audio ?gender ?genderLabel ?birthyear WHERE {
  ?person wdt:P990 ?audio .
  ?person wdt:P21 ?gender .
  FILTER (?gender IN (wd:Q6581072, wd:Q6581097))  # Avoid transgender
  OPTIONAL { ?person wdt:P569 ?birthdate .
    BIND(year(?birthdate) AS ?birthyear) }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,da" }
}
"""


def query_metadata():
    """Query Wikidata for metadata.

    This function will use the query.wikidata.org SPARQL service to query for
    items with a specified gender and which are associated with a voice audio
    file through the P990 property. Properties from Wikidata will be returned
    in columns of a Pandas dataframe.

    Returns
    -------
    metadata : pandas.DataFrame
        Dataframe with metadata.

    Examples
    --------
    >>> metadata = query_metadata()

    """
    service = sparql.Service('https://query.wikidata.org/sparql',
                             method='GET')
    response = service.query(QUERY_METADATA)
    df = DataFrame(response.fetchall(), columns=response.variables)
    return df


def save_metadata(metadata, directory=DATA_DIRECTORY):
    """Save metadata to local file.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataframe with metadata
    directory : str
        Local directory where the audio files are to be stored.

    Examples
    --------
    >>> metadata = query_metadata()
    >>> save_metadata(metadata)
    >>> metadata = load_metadata()

    """
    # Make directory if not exist
    try:
        makedirs(directory)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    filename = join(directory, METADATA_FILENAME)
    metadata.to_csv(filename, encoding='utf-8')


def load_metadata(directory=DATA_DIRECTORY):
    """Load metadata from local file.

    If the file does not exist then it is downloaded via the `query_metadata`
    function and saved for future load.

    Parameters
    ----------
    directory : str
        Local directory where the audio files are to be stored.

    Returns
    -------
    metadata : pandas.DataFrame
        Dataframe with metadata.

    """
    filename = join(directory, METADATA_FILENAME)
    try:
        metadata = read_csv(filename)
    except IOError:
        # File does not exist
        metadata = query_metadata()
        save_metadata(metadata, directory=directory)
    return metadata


def field_to_value(field):
    """Convert sparql return type to string.

    Parameters
    ----------
    field : sparql.IRI
        SPARQL return element

    Returns
    -------
    str : str
        String if 'value' attribute is present

    """
    if hasattr(field, 'value'):
        return field.value
    else:
        return field


def link_to_filename(link):
    r"""Convert Wikimedia Commons link to filename.

    The Wikimedia Commons filename is hashed withe the sha256 algorithm.
    The extension is maintained and can only be a number of restricted
    known extensions for Wikimedia Commons audio files.

    The filename is without directory information.

    Parameters
    ----------
    link : str
        String with Wikimedia Commons link

    Returns
    -------
    sha256_filename : str
        Filename part of the link with extension

    Examples
    --------
    >>> link = "http://commons.wikimedia.org/wiki/" \
    ...     "Special:FilePath/Sound.flac"
    >>> link_to_filename(link)
    'd5a0d1446b54242cd465ee0a9978e43b25ce7809c2d9d6a534fd3c7e665bac0c.flac'

    """
    link = field_to_value(link)   # possible conversion for sparql.IRI
    commons_filename = split(urlsplit(link).path, '/')[-1]
    sha256_filename = sha256(commons_filename).hexdigest()
    _, extension = splitext(link)
    if extension.lower() not in ['.flac', '.oga', '.ogg', '.wav']:
        raise Exception('Unrecognized extension: {}'.format(extension))
    filename = sha256_filename + extension
    return filename


def download_one(link, directory=DATA_DIRECTORY):
    """Download a specified Wikimedia Commons file.

    The file will be downloaded to the local data directory specified by
    DATA_DIRECTORY. The filename will be hashed from the Commons filename using
    the `link_to_filename` function.

    Parameters
    ----------
    link : str
        String with IRI
    directory : str
        Local directory where the audio files are to be stored.

    """
    link = field_to_value(link)
    filename = link_to_filename(link)
    local_filename = join(directory, filename)
    if not exists(local_filename):
        # Buffered output
        # https://gist.github.com/hughdbrown/c145b8385a2afa6570e2
        buffer = urlopen(link)
        with open(local_filename, 'wb') as output:
            while True:
                data = buffer.read(4096)
                if data:
                    # TODO: should use temporary file
                    output.write(data)
                else:
                    break


def download_all(metadata, directory=DATA_DIRECTORY):
    """Download data from Wikimedia Commons.

    This will download all audio files mentioned in the metadata. If a file is
    already downloaded it will not be downloaded anew.

    The local filename is a hashed filename determined with the
    `link_to_filename` function.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataframe with metadata
    directory : str
        Local directory where the audio files are to be stored.

    Examples
    --------
    >>> download_all()

    """
    # Make directory if not exist
    try:
        makedirs(directory)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    # Download audio files iteratively
    for remote_filename in metadata.audio:
        download_one(remote_filename, directory=directory)


def iter_filenames(metadata, directory=DATA_DIRECTORY, yield_gender=False):
    """Yield filenames for audio files.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataframe with metadata.
    directory : str
        Local directory where the audio files are to be stored.
    yield_gender : bool
        Also yield gender if True.

    Yields
    ------
    filename : str
        Local filename for audio file
    gender : str
        Gender represented as string either 'male' or 'female'

    """
    for index, row in metadata.iterrows():
        remote_filename = field_to_value(row.audio)
        filename = link_to_filename(remote_filename)
        local_filename = join(directory, filename)
        gender = field_to_value(row.genderLabel)
        if yield_gender:
            yield local_filename, gender
        else:
            yield local_filename


def get_filenames(metadata, directory=DATA_DIRECTORY):
    """Return list of filenames.

    Returns
    -------
    filenames : list of str
        List of filenames with full path.

    """
    return list(iter_filenames(metadata, directory=directory))


def iter_audio_segments(metadata, directory=DATA_DIRECTORY,
                        yield_gender=False):
    """Yield audio segments objects.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataframe with metadata
    directory : str
        Local directory where the audio files are to be stored.
    yield_gender : bool
        Also yield gender if True.

    Yields
    ------
    audio_segment : pydub.AudioSegment
        Audio segments.
    gender : str
        Gender represented as URI

    """
    for filename, gender in iter_filenames(
            metadata, directory=directory, yield_gender=True):
        try:
            audio_segment = AudioSegment.from_file(filename)
        except wave.Error:
            # The 'wave' module cannot read certain wave files.
            # Here these files are ignored.
            continue
        if yield_gender:
            yield audio_segment, gender
        else:
            yield audio_segment


def get_audio_segment(filename, directory=DATA_DIRECTORY):
    """Return audio segment.

    Parameters
    ----------
    filename : str
        Full path filename to audio file.

    Returns
    -------
    audio_segment : AudioSegment
        Object with audio data.

    """
    audio_segment = AudioSegment.from_file(filename)
    return audio_segment


def iter_audio_segments_mono_11025(
        metadata, directory=DATA_DIRECTORY, yield_gender=False):
    """Yield audio segments in mono and 11025 Hz.

    The first channel is returned after a split to mono.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataframe with metadata
    directory : str
        Local directory where the audio files are to be stored.
    yield_gender : bool
        Also yield gender if True.

    Yields
    ------
    audio_segment : pydub.AudioSegment
        Resampled audio segments in mono.
    gender : str
        URI representing gender.

    Examples
    --------
    >>> metadata = load_metadata()
    >>> audio_segments = audio_segments_mono_11025(metadata)
    >>> audio_segment = audio_segments.next()
    >>> audio_segment.channels
    1
    >>> audio_segment.frame_rate
    11025

    >>> audio_segments = iter_audio_segments_mono_11025(
    ...     metadata, yield_gender=True)
    >>> audio_segment, gender = audio_segments.next()
    >>> gender in ['male', 'female']
    True

    """
    for audio_segment, gender in iter_audio_segments(
            metadata, directory=directory, yield_gender=True):
        output = audio_segment.split_to_mono()[0].set_frame_rate(11025)
        if yield_gender:
            yield output, gender
        else:
            yield output


def get_audio_segment_mono_11025(audio, directory=DATA_DIRECTORY):
    """Convert audio segment to mono and 11025 Hertz sample rate.

    Parameters
    ----------
    audio : str or pydub.AudioSegment
        Audio segment and AudioSegment or filename
    directory : str
        Local directory where the audio files are to be stored.

    Returns
    -------
    audio_segment : pydub.AudioSegment
        Audio segment

    """
    if type(audio) == str:
        filename = audio
        audio_segment = get_audio_segment(filename, directory=directory)
    elif type(audio) == AudioSegment:
        audio_segment = audio
    else:
        raise ValueError('audio input has the wrong type')
    if audio_segment.channels == 1 and audio_segment.frame_rate == 11025:
        output = audio_segment
    else:
        output = audio_segment.split_to_mono()[0].set_frame_rate(11025)
    return output


def iter_samples_mono_11025(
        metadata, directory=DATA_DIRECTORY, yield_gender=False):
    """Yield samples in mono and 11025 Hz.

    The first channel is returned after a split to mono.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataframe with metadata
    directory : str
        Local directory where the audio files are to be stored.
    yield_gender : bool
        Also yield gender if True.

    Yields
    ------
    samples : numpy.array
        Resampled audio segments in mono as numpy array
    gender : str
        Gender represented as string

    Examples
    --------
    >>> metadata = load_metadata()
    >>> samples = iter_samples_mono_11025(metadata, yield_gender=True)
    >>> sample, gender = samples.next()

    """
    for audio_segment, category in iter_audio_segments_mono_11025(
            metadata, directory=directory, yield_gender=True):
        output = np.array(audio_segment.get_array_of_samples())
        if yield_gender:
            yield output, category
        else:
            yield output


def get_samples_mono_11025(
        audio, directory=DATA_DIRECTORY, yield_gender=False):
    """Return samples in mono and 11025 Hz.

    The first channel is returned after a split to mono.

    Parameters
    ----------
    audio : str or pydub.AudioSegment
        Filename of audio object
    directory : str
        Local directory where the audio files are to be stored.
    yield_gender : bool
        Also yield gender if True.

    Returnes
    --------
    samples : numpy.array
        Resampled audio segments in mono as numpy array

    """
    audio_segment = get_audio_segment_mono_11025(audio)
    output = np.array(audio_segment.get_array_of_samples())
    return output


def iter_chunk(samples, chunk_size=1024):
    """Yield chunks.

    Generator yield Numpy array chunks from a longer Numpy array.

    Parameters
    ----------
    samples : array_like
        List of samples in numpy array.
    chunk_size : int
        Number of samples in each yielded chunk.

    Yields
    ------
    sample_chunk : array_like
        List of samples in numpy array.

    """
    n_chunks = ((samples.shape[0] - 1) // chunk_size) + 1
    for n in iter(range(n_chunks)):
        offset = n * chunk_size
        end = min(samples.shape[0], offset + chunk_size)
        sample_chunk = samples[offset:end]
        yield sample_chunk


def iter_samples_to_pitches(
        samples, sample_rate=11025, method='yin', buf_size=2048, hop_size=256,
        yield_confidence=False):
    """Yield pitches.

    Parameters
    ----------
    samples : numpy.ndarray
        Samples in Numpy array.

    Yields
    ------
    pitch : float
        Pitch in Hertz
    confidence : float
        Confidence of the pitch detection.

    """
    pitcher = aubio.pitch(method, buf_size, hop_size, int(sample_rate))
    for sample_chunk in iter_chunk(samples):
        pitch = pitcher(sample_chunk.astype(np.float32))[0]
        if yield_confidence:
            confidence = pitcher.get_confidence()
            yield pitch, confidence
        else:
            yield pitch


def get_pitches(audio, sample_rate=11025, method='yin', buf_size=2048,
                hop_size=256):
    """Return estimated pitches for audio.

    The 'yin' method from the `aubio` module is used for the pitch detection.
    Multiple values are returned, one for each chunk.

    Parameters
    ----------
    audio : numpy.array or pydub.AudioSegment or string
        Audio data.

    Returns
    -------
    pitches : numpy.array
        Array with pitch and confidence information.

    """
    if type(audio) == np.ndarray:
        samples = audio
    elif type(audio) == AudioSegment or type(audio) == str:
        samples = get_samples_mono_11025(audio)
        sample_rate = 11025
    else:
        raise ValueError('audio input has the wrong type')

    pitcher = aubio.pitch(method, buf_size, hop_size, int(sample_rate))

    # Iterate over chunks in the sample
    pitches = []
    for sample_chunk in iter_chunk(samples):
        pitch = pitcher(sample_chunk.astype(np.float32))[0]
        confidence = pitcher.get_confidence()
        pitches.append((pitch, confidence))
    return np.array(pitches)


def get_pitch(audio, sample_rate=11025, method='yin', buf_size=2048,
              hop_size=256):
    """Return estimated pitch for audio.

    Parameters
    ----------
    audio : samples or audio segment or filename.
        Audio data.
    sample_rate : float or int
        Sample rate in Hertz. Should be specified to the correct value for
        samples input, otherwise inferred from the audio segment or file.

    Returns
    -------
    pitch : float
        Single value for pitch.

    """
    if type(audio) == np.ndarray:
        pitches = get_pitches(
            audio, sample_rate=sample_rate, method=method, buf_size=buf_size,
            hop_size=hop_size)
    else:
        pitches = get_pitches(
            audio, method=method, buf_size=buf_size, hop_size=hop_size)

    # Top confidences
    indices = np.argsort(-pitches[:, 1])[:min(10, pitches.shape[0])]

    pitch = np.median(pitches[indices, 0])

    return pitch


def detect_gender(audio, sample_rate=11025, method='yin', buf_size=2048,
                  hop_size=256): 
    """Detect gender of audio.

    The present detection method is based on pitch detection and a threshold.

    Parameters
    ----------
    audio : samples or audio segment or filename.
        Audio data.
    sample_rate : float or int
        Sample rate in Hertz. Should be specified to the correct value with the
        audio input is samples. Otherwise it is inferred from the audio segment
        or file.

    Returns
    -------
    gender : str
        String representing gender as 'male' or 'female'.

    """
    if type(audio) == np.ndarray:
        pitch = get_pitch(audio, sample_rate=sample_rate, method=method,
                          buf_size=buf_size, hop_size=hop_size)
    else:
        pitch = get_pitch(audio, method=method, buf_size=buf_size,
                          hop_size=hop_size)

    if pitch > 145:
        return 'female'
    else:
        return 'male'


def iter_capture_and_detect_gender(sample_rate = 11025, hop_size=256,
                                   confidence_threshold=0.8):
    """Capture audio and yield gender.

    Parameters
    ----------
    sample_rate : int
        Sample rate in Hertz
    confidence_threshold : float
        Pitch confidence threshold for yielding

    Yields
    ------
    gender : str
        String representing gender either 'male' or 'female'

    """
    stream = Stream(blocksize=hop_size, channels=1, samplerate=sample_rate)
    stream.start()
    try:
        while True:
            samples = stream.read(hop_size).flatten()
            pitch_and_confidence = get_pitches(
                samples, buf_size=hop_size, hop_size=hop_size,
                sample_rate=sample_rate)
            pitch = pitch_and_confidence[0, 0]
            confidence = pitch_and_confidence[0, 1]
            if pitch > 145:
                gender = 'female'
            else:
                gender = 'male'
            if confidence >= confidence_threshold:
                yield gender
    except KeyboardInterrupt:
        pass
    stream.stop()
