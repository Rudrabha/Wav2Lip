#!/usr/bin/env bash

NAME=wav2lip-dev

set -ex

docker build . -t $NAME
docker run -it --rm \
  --name $NAME \
  -v $PWD/checkpoints:/src/checkpoints \
  -p 6001:5000 \
  --gpus all \
  $NAME
