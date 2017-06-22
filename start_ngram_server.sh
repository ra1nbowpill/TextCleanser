#!/bin/bash

[[ $1 =~ ^-h$ ]] && echo "Usage: $0 [port]" && exit

# $1 is the port to be used
SERVER_PORT="12345"
[[ $1 =~ ^[0-9]{1,5}$ ]] && SERVER_PORT="$1"


# update with your own paths

# LATTICE_TOOL_DIR=/media/Data/work/Tools/srilm/bin/i686-m64
LATTICE_TOOL_DIR="/Users/ygorgallina/Documents/Cours/M1/Stage/Outils/srilm-1.6.0/bin/macosx/"

# LM_DIR=/media/Data/work/Development/EclipseWorkspace/TextCleanser/data/
LM_DIR="/Users/ygorgallina/Documents/Cours/M1/Stage/Outils/TextCleanser/data/"

$LATTICE_TOOL_DIR/ngram -lm "$LM_DIR/tweet-lm.gz" -mix-lm "$LM_DIR/latimes-lm.gz" -lambda 0.7 -mix-lambda2 0.3 -server-port "$SERVER_PORT" &
echo "$!"