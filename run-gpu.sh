#!/bin/sh
docker run --gpus all --rm \
  -p 8888:8888 \
  tensorfi/tensorfi:gpu
