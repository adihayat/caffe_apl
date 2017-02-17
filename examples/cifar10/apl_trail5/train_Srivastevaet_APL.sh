#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe test \
    --solver examples/cifar10/apl_trail5/cifar10_Srivastavaet_APL_solver.prototxt \
    |& tee examples/cifar10/apl_trail5/train.log 
