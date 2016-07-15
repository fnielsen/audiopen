from __future__ import print_function

from audiopen.gendervoice import iter_capture_and_detect_gender

for gender in iter_capture_and_detect_gender():
    print(gender)
