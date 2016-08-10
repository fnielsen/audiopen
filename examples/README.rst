Examples
========



capture_and_detect_gender.py
----------------------------

This script will attempt to open a real-time audio-channel, capture its data and output the detected gender.
Only when the pitch detection algorithm is sufficient confident the script will report the gender.

A session might look like this:

.. code:: bash

    $ python capture_and_detect_gender.py 
    ALSA lib pcm_dsnoop.c:606:(snd_pcm_dsnoop_open) unable to open slave
    ALSA lib pcm_dmix.c:1029:(snd_pcm_dmix_open) unable to open slave
    ALSA lib pcm.c:2266:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.rear
    ALSA lib pcm.c:2266:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.center_lfe
    ALSA lib pcm.c:2266:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.side
    ALSA lib pcm_dmix.c:1029:(snd_pcm_dmix_open) unable to open slave
    Cannot connect to server socket err = No such file or directory
    Cannot connect to server request channel
    jack server is not running or cannot be started
    JackShmReadWritePtr::~JackShmReadWritePtr - Init not done for 4294967295, skipping unlock
    JackShmReadWritePtr::~JackShmReadWritePtr - Init not done for 4294967295, skipping unlock
    /usr/local/lib/python2.7/dist-packages/cffi/model.py:526: UserWarning: 'enum PaHostApiTypeId' has no values explicitly defined; next version will refuse to guess which integer type it is meant to be (unsigned/signed, int/long)
      % self._get_c_name())
    male
    male
    male
    male
    male
    male
    male
    female
    female
    female
    female
    female


gender_pitches.py
-----------------

Pitch detection of audio data from Wikimedia Commons.

.. code:: bash

    $ python gender_pitches.py 
    ALSA lib pcm.c:2266:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.rear
    ALSA lib pcm.c:2266:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.center_lfe
    ALSA lib pcm.c:2266:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.side
    ALSA lib pcm_route.c:867:(find_matching_chmap) Found no matching channel map
    ALSA lib pcm_route.c:867:(find_matching_chmap) Found no matching channel map
    ALSA lib pcm_route.c:867:(find_matching_chmap) Found no matching channel map
    ALSA lib pcm_route.c:867:(find_matching_chmap) Found no matching channel map
    Cannot connect to server socket err = No such file or directory
    Cannot connect to server request channel
    jack server is not running or cannot be started
    JackShmReadWritePtr::~JackShmReadWritePtr - Init not done for 4294967295, skipping unlock
    JackShmReadWritePtr::~JackShmReadWritePtr - Init not done for 4294967295, skipping unlock
    /home/faan/data/audiopen/3037f71cbd844a2373c3a3252cac2abd03e5c0f10dd12a969c3604ef97f2826e.oga
    male    - 136.12 Hz - Igor Czubajs
    /home/faan/data/audiopen/d72015ce1512e1c524b3f42dbba60b0253a4ffb93f4bfa74ceee55ee770d69b7.oga
    male    - 105.88 Hz - Alexander Shokhin
    /home/faan/data/audiopen/deb9d71559e99586c9eb254e841e503e3664b244aecf6fbc38d6f70fe8f179ac.oga
    male    - 144.01 Hz - Dmitry Shparo
    /home/faan/data/audiopen/deb9d71559e99586c9eb254e841e503e3664b244aecf6fbc38d6f70fe8f179ac.oga
    male    - 144.01 Hz - Dmitry Shparo
    /home/faan/data/audiopen/ee4111947aef9d05dd915bb825b27c5488febc7caac844914a029fecf97d61ff.ogg
    female  - 112.31 Hz - Alice Arnold
    /home/faan/data/audiopen/03497cf1b087feb14ad61d8fb67c725d258e23bf1f1782d612e4193efa46a48e.flac
    male    - 93.41 Hz - Andrew Hussey


features.py
-----------

.. code:: bash


    $ python features.py
    [-10.1418314   21.7928257  -16.97315979  19.26405144 -17.00856018
      17.58937836 -15.83039188  15.42481899 -14.28900909  13.13132477
     -12.09650803  10.68149757  -9.83863544  42.81553268   0.        ]
    [ -2.46730995e+00   2.32820225e+01  -1.72524529e+01   1.98781567e+01
      -1.73932934e+01   1.80337982e+01
      ...
