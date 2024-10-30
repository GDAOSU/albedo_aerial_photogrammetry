#!/bin/bash
if [ -z "$workdir" ]; then
    echo "workdir is empty"
    exit 1
fi

# get folder of current script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

$DIR/../install-Release/bin/ambientocc_baker --modeldataset=$workdir/modeldataset.json --output=$workdir --use_sphere=false --num_samples=512 --epsilon=0.1 --radius=20 -v 0
