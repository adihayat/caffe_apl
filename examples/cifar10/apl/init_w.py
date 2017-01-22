import caffe
print "Loading Net"
net = caffe.Net('cifar10_Srivastavaet_APL_deploy.prototxt',caffe.TEST)
for L in net.layers:
    if L.type == "APL":
        L.blobs[0].data[0,0,0,:] = -0.01
        L.blobs[0].data[0,0,1,:] = -0.001
        L.blobs[1].data[0,0,1,:] = -1

print "Saving Net"
net.save('initialized.caffemodel')
print "Done initalization"
