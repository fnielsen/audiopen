"""Handle data."""

import errno

from os import makedirs
from os.path import exists, expanduser, join, split

from urllib2 import urlopen

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
    
    # Download audio files
    metadata = read_metadata()
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
