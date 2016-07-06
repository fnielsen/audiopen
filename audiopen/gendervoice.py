"""Gendervoice."""

import errno

from os import makedirs
from os.path import exists, expanduser, join

from string import split

from urllib2 import urlopen

from urlparse import urlsplit

from pandas import DataFrame

import sparql


DATA_DIRECTORY = join(expanduser('~'), 'data', 'audiopen')

QUERY_METADATA = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

select ?person ?personLabel ?audio ?gender ?genderLabel where {
  ?person wdt:P990 ?audio .
  ?person wdt:P21 ?gender .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,da" }
}
"""


def query_metadata():
    """Query Wikidata for metadata.

    This function will use the query.wikidata.org SPARQL service.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe

    """
    service = sparql.Service('https://query.wikidata.org/sparql')
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

    Parameters
    ----------
    link : str
        String with Wikimedia Commons link

    Returns
    -------
    filename : str
        Filename part of the link

    Examples
    --------
    >>> link = "http://commons.wikimedia.org/wiki/" \
    ...     "Special:FilePath/Sound.flac"
    >>> link_to_filename(link)
    'Sound.flac'

    """
    link = field_to_value(link)   # possible conversion for sparql.IRI
    filename = split(urlsplit(link).path, '/')[-1]
    return filename


def download_one(link, directory=DATA_DIRECTORY):
    """Download a specified Wikimedia Commons file.

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

    This will download all audio files mentioned in the metadata,
    if they are not already downloaded.

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
