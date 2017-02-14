import numpy as np
from matplotlib import pyplot as plt
import caffe
import cv2
net = caffe.Net('examples/cifar10/apl_trail4/cifar10_Srivastavaet_APL_deploy.prototxt','examples/cifar10/apl_trail4/snapshot/_iter_60000.caffemodel',caffe.TEST)
net_i = caffe.Net('examples/cifar10/apl_trail4/cifar10_Srivastavaet_APL_deploy.prototxt','examples/cifar10/apl_trail4/initialized.caffemodel',caffe.TEST)
net_b = caffe.Net('examples/cifar10/apl_trail4/cifar10_Srivastavaet_APL_deploy.prototxt','examples/cifar10/apl_trail4/snapshot/_iter_30000.caffemodel',caffe.TEST)
#net_b = caffe.Net('examples/cifar10/apl_trail3/cifar10_Srivastavaet_APL_deploy.prototxt','examples/cifar10/apl_trail3/snapshot/_iter_60000.caffemodel',caffe.TEST)

x = np.arange(-1,1,0.01)
ys = []
ys_i = []
ys_b = []
offset = 1000
for p in xrange(1,4):
    ys.append([])
    ys_i.append([])
    ys_b.append([])
    str_p = str(p)
    apl1_w = net.params['apl'+str_p][0].data
    apl1_b = net.params['apl'+str_p][1].data
    apl_i_w = net_i.params['apl'+str_p][0].data
    apl_i_b = net_i.params['apl'+str_p][1].data
    apl_b_w = net_b.params['apl'+str_p][0].data
    apl_b_b = net_b.params['apl'+str_p][1].data
    for n in xrange(offset,offset + 5):
        y = np.maximum(x,0)
        for ind in xrange(apl1_b.shape[2]):
            y += np.maximum(-1*x + apl1_b[0,0,ind,n],0)*apl1_b[0,0,ind,n]

        ys[-1].append(y)

        y = np.maximum(x,0)
        for ind in xrange(apl1_b.shape[2]):
            y += np.maximum(-1*x + apl_i_b[0,0,ind,n],0)*apl_i_w[0,0,ind,n]

        ys_i[-1].append(y)

        y = np.maximum(x,0)
        for ind in xrange(apl1_b.shape[2]):
            y += np.maximum(-1*x + apl_b_b[0,0,ind,n],0)*apl_b_w[0,0,ind,n]

        ys_b[-1].append(y)


f , axarr = plt.subplots(3,5,figsize=(15,15))
for l_idx,(y_p,y_i_p,y_b_p) in enumerate(zip(ys,ys_i,ys_b)):
    for p_idx,(y,y_i,y_b) in enumerate(zip(y_p,y_i_p,y_b_p)):
        axarr[l_idx,p_idx].plot(x,y,x,y_i,'--',x,y_b,'--',linewidth=1.5)
        if l_idx == 0:
            axarr[l_idx,p_idx].set_title('pixel {}'.format(100+p_idx))
    axarr[l_idx,0].set_ylabel('Apl {}'.format(l_idx))


plt.show()
#plt.savefig('apl.png')
