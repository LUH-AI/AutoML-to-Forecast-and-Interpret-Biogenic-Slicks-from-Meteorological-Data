#!/bin/bash
# Start the Ray head node.

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "starting ray head node"
ray start --head \
    --node-ip-address="$1" \
    --port="$2" \
    --dashboard-port="$3" \
    --temp-dir="$4"
sleep infinity
