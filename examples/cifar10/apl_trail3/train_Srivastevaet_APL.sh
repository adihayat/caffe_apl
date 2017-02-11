#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver examples/cifar10/apl_trail3/cifar10_Srivastavaet_APL_solver.prototxt \
    --weights examples/cifar10/apl_trail3/initialized.caffemodel \
    |& tee examples/cifar10/apl_trail3/train.log 
