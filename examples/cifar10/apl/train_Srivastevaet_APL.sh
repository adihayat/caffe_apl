#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver examples/cifar10/apl/cifar10_Srivastavaet_APL_solver.prototxt \
    --weights examples/cifar10/apl/initialized.caffemodel \
    |& tee examples/cifar10/apl/train.log 
