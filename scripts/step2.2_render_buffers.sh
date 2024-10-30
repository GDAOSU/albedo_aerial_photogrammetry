#!/bin/bash
if [ -z "$workdir" ]; then
    echo "workdir is empty"
    exit 1
fi

set nLevel=$1
if [ -z "$nLevel" ]; then
    nLevel=2
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

$DIR/../install-Release/bin/tinyrender --imagedataset=$workdir/imagedataset.json --modeldataset=$workdir/modeldataset.json --output=$workdir --cull --normal --skyvis --sunvis --level=$nLevel -v 2
