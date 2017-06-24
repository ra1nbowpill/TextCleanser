#!/usr/bin/python

"""
    A command line accessor for the TextCleanser. Accepts noisy strings
    as input on stdin and outputs normalised strings on stdout.

    Author: Stephan Gouws
    Contact: stephan@ml.sun.ac.za
"""

import sys
from cleanser import TextCleanser

if __name__ == '__main__':
    clnsr = TextCleanser()

    text = sys.stdin.readline()
    while text:
        if len(text) <= 1:
            break

        cleantext, error, replacements = clnsr.ibm_cleanse(text, gen_off_by_ones=False)

        if error == "ERROR":
            sys.stderr.write("ERROR")
            continue

        else:
            sys.stdout.write(cleantext)

        # need to flush the output buffers so that the java wrapper can read in
        # the input
        sys.stdout.flush()
        sys.stderr.flush()
        text = sys.stdin.readline()
