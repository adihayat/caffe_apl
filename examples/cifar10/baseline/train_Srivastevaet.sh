#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver examples/cifar10/baseline/cifar10_Srivastavaet_solver_4.prototxt \
    --weights examples/cifar10/baseline/cifar10_Srivastavaet_iter_60000.caffemodel.h5 \
    |& tee examples/cifar10/baseline/cifar10_Srivastavaet.log 
