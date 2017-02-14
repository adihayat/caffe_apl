import caffe
import numpy as np
print "Loading Net"
net = caffe.Net('cifar10_Srivastavaet_APL_deploy.prototxt',caffe.TEST)
for L in net.layers:
    if L.type == "APL":
        L.blobs[0].data[0,0,0,:] =  0.5   + np.random.uniform(-0.05,0.05,L.blobs[0].data[0,0,0,:].shape)
        L.blobs[1].data[0,0,0,:] =  0.35  + np.random.uniform(-0.05,0.05,L.blobs[0].data[0,0,0,:].shape)
        L.blobs[0].data[0,0,1,:] = -0.7  + np.random.uniform(-0.05,0.05,L.blobs[0].data[0,0,0,:].shape)
        L.blobs[1].data[0,0,1,:] =  0
        L.blobs[0].data[0,0,2,:] =  0.15  + np.random.uniform(-0.05,0.05,L.blobs[0].data[0,0,0,:].shape)
        L.blobs[1].data[0,0,2,:] = -0.35   + np.random.uniform(-0.05,0.05,L.blobs[0].data[0,0,0,:].shape)

print "Saving Net"
net.save('initialized.caffemodel')
print "Done initalization"

