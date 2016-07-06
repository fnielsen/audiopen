"""Gendervoice.

"""

from string import split

from os.path import expanduser, join

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

    Returns
    -------
    df : pandas.DataFrame
        Dataframe

    """
    service = sparql.Service('https://query.wikidata.org/sparql')
    response = service.query(QUERY_METADATA)
    df = DataFrame(response.fetchall(), columns=response.variables)
    return df


def link_to_filename(link):
    """Convert Wikimedia Commons link to filename.

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
    filename = split(urlsplit(link).path, '/')[-1]
    return filename
