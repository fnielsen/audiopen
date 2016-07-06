"""Audiopen.

Examples
--------
>>> import audiopen.data

>>> # Download audio files from Wikimedia Commons
>>> audiopen.data.download()

"""

from __future__ import absolute_import

from .data import download
from .metadata import read_metadata
