#!/bin/bash
# Start a Ray worker node. Called by run_array.sh with:
#   $1 head address as ip:port, $2 temp dir

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "starting ray worker node"
ray start --address $1 --temp-dir=$2
sleep infinity
