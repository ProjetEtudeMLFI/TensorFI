#!/bin/sh
docker build . --rm -f Dockerfile.gpu -t tensorfi/tensorfi:gpu
