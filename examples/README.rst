Examples
========



capture_and_detect_gender.py
----------------------------

This script will attempt to open a real-time audio-channel, capture its data and output the detected gender.
Only when the pitch detection algorithm is sufficient confident the script will report the gender.

A session might look like this:

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
