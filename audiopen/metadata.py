"""Handles to metadata."""


from os.path import join, split

from pandas import read_csv


def read_metadata():
    """Read metadata data.

    Returns
    -------
    metadata : pandas.DataFrame
        Dataframe with metadata

    Examples
    --------
    >>> metadata = read_metadata()
    >>> 'music' in metadata.Category.values
    True

    >>> 'Commons' in metadata.columns
    True

    """
    path, _ = split(__file__)
    filename = join(path, 'metadata', 'audiopen.csv')
    metadata = read_csv(filename)
    return metadata
