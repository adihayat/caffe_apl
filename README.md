# DL Seminar project
this code tests and reproduces https://arxiv.org/abs/1412.6830
Learning Activation Functions to Improve Deep Neural Networks
by Forest Agostinelli, Matthew Hoffman, Peter Sadowski, Pierre Baldi

in this code:
- runme_train.py : reproduces the article main result 
this code makes sure that the right caffe version is built, downloads the cifar10 dataset and trains the net.
to disable the downloading of the cifar10 dataset , just comment out the "getCifar10" function
- runme_test.py <weights_file> : measures the trained net accuracy , receives as input the weight file provided in the dropbox link 


some more expirements files are available at : `examples/cifar10/apl*`




