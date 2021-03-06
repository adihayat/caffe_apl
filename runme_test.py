import os
import sys


def check_caffe_build():
    if os.system('./build/tools/caffe') !=0 :
        print "Error: caffe was not built , please do make ; make pycaffe"
        sys.exit(1)


def getCifar10_Data():
    if os.system('./data/cifar10/get_cifar10.sh')!=0:
        print "Error was unable to download cifar10 dataset"
        sys.exit(1)

def createCifar10_Data():
    if os.system('./examples/cifar10/create_cifar10.sh')!=0:
        print "Error was unable to create cifar10 dataset"
        sys.exit(1)

def run_test():
    if os.system('bash ./examples/cifar10/apl_trail5/test_Srivastevaet_APL.sh ' + sys.argv[1])!=0:
        print "Error was unable to run test"
        sys.exit(1)



check_caffe_build()
getCifar10_Data()
createCifar10_Data()
run_test()
