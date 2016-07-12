"""Gendervoice.

Binary gender is represented with Wikidata URI: 

http://www.wikidata.org/entity/Q6581097 is man

Examples
--------
>>> import audiopen.gendervoice
>>> metadata = audiopen.gendervoice.query_metadata()

"""

import errno

from hashlib import sha256

from os import makedirs
from os.path import exists, expanduser, join, splitext

from string import split

from urllib2 import urlopen

from urlparse import urlsplit

import aubio

import numpy as np

from pandas import DataFrame

from pydub import AudioSegment

import sparql


DATA_DIRECTORY = join(expanduser('~'), 'data', 'audiopen')

QUERY_METADATA = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>

SELECT ?person ?personLabel ?audio ?gender ?genderLabel WHERE {
  ?person wdt:P990 ?audio .
  ?person wdt:P21 ?gender .
  FILTER (?gender IN (wd:Q6581072, wd:Q6581097))  # Avoid transgender
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,da" }
}
"""


def query_metadata():
    """Query Wikidata for metadata.

    This function will use the query.wikidata.org SPARQL service to query for
    items with a specified gender and which are associated with a voice audio
    file through the P990 property.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe

    Examples
    --------
    >>> metadata = query_metadata()

    """
    service = sparql.Service('https://query.wikidata.org/sparql',
                             method='GET')
    response = service.query(QUERY_METADATA)
    df = DataFrame(response.fetchall(), columns=response.variables)
    return df


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


def filenames(metadata, directory=DATA_DIRECTORY, yieldgender=False):
    """Yield filenames for audio files.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataframe with metadata.
    directory : str
        Local directory where the audio files are to be stored.
    yieldgender : bool
        Also yield gender if True.

    Yields
    ------
    filename : str
        Local filename for audio file
    gender : str
        Gender represented as URI

    """
    for index, row in metadata.iterrows():
        remote_filename = field_to_value(row.audio)
        filename = link_to_filename(remote_filename)
        local_filename = join(directory, filename)
        gender = field_to_value(row.gender)
        if yieldgender:
            yield local_filename, gender
        else:
            yield local_filename


def audio_segments(metadata, directory=DATA_DIRECTORY, yieldgender=False):
    """Yield audio segments objects.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataframe with metadata
    directory : str
        Local directory where the audio files are to be stored.
    yieldgender : bool
        Also yield gender if True.

    Yields
    ------
    audio_segment : pydub.AudioSegment
        Audio segments.

    """
    for filename, gender in filenames(
            metadata, directory=directory, yieldgender=True):
        audio_segment = AudioSegment.from_file(filename)
        if yieldgender:
            yield audio_segment, gender
        else:
            yield audio_segment


def audio_segments_mono_11025(
        metadata, directory=DATA_DIRECTORY, yieldgender=False):
    """Yield audio segments in mono and 11025 Hz.

    The first channel is returned after a split to mono.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataframe with metadata
    directory : str
        Local directory where the audio files are to be stored.
    yieldgender : bool
        Also yield gender if True.

    Yields
    ------
    audio_segment : pydub.AudioSegment
        Resampled audio segments in mono.
    gender : str
        URI representing gender.

    Examples
    --------
    >>> metadata = query_metadata()
    >>> audio_segments = audio_segments_mono_11025(metadata)
    >>> audio_segment = audio_segments.next()
    >>> audio_segment.channels
    1
    >>> audio_segment.frame_rate
    11025

    >>> audio_segments = audio_segments_mono_11025(metadata, yieldgender=True)
    >>> audio_segment, gender = audio_segments.next()
    >>> gender in ['https://www.wikidata.org/wiki/Q6581097',
    ...     'https://www.wikidata.org/wiki/Q6581072']
    True

    """
    for audio_segment, gender in audio_segments(
            metadata, directory=directory, yieldgender=True):
        output = audio_segment.split_to_mono()[0].set_frame_rate(11025)
        if yieldgender:
            yield output, gender
        else:
            yield output


def samples_mono_11025(
        metadata, directory=DATA_DIRECTORY, yieldgender=False):
    """Yield audio segments in mono and 11025 Hz.

    The first channel is returned after a split to mono.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataframe with metadata
    directory : str
        Local directory where the audio files are to be stored.
    yieldgender : bool
        Also yield gender if True.

    Yields
    ------
    samples : numpy.array
        Resampled audio segments in mono as numpy array
    gender : str
        Gender of sample represented as a string. It is only yielded if
        yieldgender is True

    Examples
    --------
    >>> metadata = query_metadata()
    >>> samples = samples_mono_11025(metadata, yieldgender=True)
    >>> sample, gender = samples.next()

    """
    for audio_segment, category in audio_segments_mono_11025(
            metadata, directory=directory, yieldgender=True):
        output = np.array(audio_segment.get_array_of_samples())
        if yieldgender:
            yield output, category
        else:
            yield output


def chunk(samples, chunksize=1024):
    """Yield chunks."""
    n_chunks = (samples.shape[0] // chunksize) + 1
    for n in iter(range(n_chunks)):
        offset = n * chunksize
        end = min(samples.shape[0], offset + chunksize)
        sample_chunk = samples[offset:end]
        yield sample_chunk


def samples_to_pitches(
        samples, method='yin', bufsize=2048, hopsize=256, samplerate=11025):
    """Yield pitches."""
    pitcher = aubio.pitch(method, bufsize, hopsize, samplerate)
    for sample_chunk in chunk(samples):
        pitch = pitcher(sample_chunk.astype(np.float32))[0]
        confidence = pitcher.get_confidence()
        yield pitch
