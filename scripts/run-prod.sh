#!/usr/bin/env bash

NAME=wav2lip

set -x

docker rm -f $NAME

docker build . -t $NAME
docker run -d --restart always \
  --name $NAME \
  -v $PWD/checkpoints:/src/checkpoints \
  -p 5001:5000 \
  --gpus all \
  $NAME

docker logs -f $NAME
