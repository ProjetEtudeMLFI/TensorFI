#!/bin/sh
docker build . --rm -f Dockerfile.cpu -t tensorfi/tensorfi:cpu
