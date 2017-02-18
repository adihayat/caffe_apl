#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe test \
    --model examples/cifar10/apl_trail5/cifar10_Srivastavaet_APL_train_test.prototxt \
    --weights $1  \
    --iterations 100
