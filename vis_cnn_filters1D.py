#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys

# Make sure that caffe is on the python path:
#caffe_root = '/home/brt/code/caffe/'  # this file is expected to be in {caffe_root}/examples
#sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=0, padval=0):
    data -= data.min()
    if data.max != 0.0:
        data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    print 'Data_shape = ', data.shape

    #plt.imshow(data)
    plt.plot(data)
    plt.waitforbuttonpress()
    
# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_time_filters(data, padsize=0, padval=0):
    #Just some normalization    
    data -= data.min()
    if data.max != 0.0:
        data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    max_indx = data.shape[0]    
    print 'Data shape = ', data.shape    
    
    for row_i in range(n):
        for col_i in range(n):
            indx = row_i*n + col_i
            if indx >= max_indx:
                break
            graph = np.array( data[indx][0] ) 
            plt.subplot(n,n,indx)
            print 'Graph = ', graph, ' Shape = ', graph.shape
            plt.plot(graph)
    
    plt.waitforbuttonpress()

def netInitialization(netProtoFile, modelFile, meanFile=[]):

    #NOTE : Not Working, throwing error
    # caffe.set_phase_test()
    # caffe.set_mode_cpu()

    net = caffe.Classifier(netProtoFile,
                        modelFile)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    #print 'MeanFile : ' + meanFile
    if(meanFile != []):
        net.set_mean('data', np.load(meanFile))
    # net.set_raw_scale('data', 255)
    # net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    return net

def predictAndVisualize(net, inputImg):

    netBlobs = [(k, v.data.shape) for k, v in net.blobs.items()]
    netParams = [(k, v[0].data.shape) for k, v in net.params.items()]

    #scores = net.predict([caffe.io.load_image(inputImg)])
    #print scores
    # the parameters are a list of [weights, biases]

    # plt.imshow(net.deprocess('data', net.blobs['data'].data[4]))

    plt.figure()
    filtersConv1 = net.params['conv0'][0].data
    print 'Filters blob shape = ', filtersConv1.shape
    # Change data format from NumOfItmesxDimsxHeightxWidth to NumOfItemsxHeightxWidthxDims
    #vis_square(filtersConv1.transpose(0, 2, 3, 1))
    vis_time_filters(filtersConv1)
    plt.draw()

    #plt.figure()
    #feat = net.blobs['conv1'].data[0, :36]
    #vis_square(feat, padval=1)

    #plt.figure()
    #filtersConv2 = net.params['conv1'][0].data
    #vis_square(filtersConv2[:48].reshape(48**2, 5, 5))


if __name__== '__main__':

    print "Inside Here"



    if(len(sys.argv) < 4):
        print     'Arguments:\n   #1st - model description (prototxt)\n   #2nd - trained model (snapshot)\n   #3rd - output image file with visualization'
        exit()
        
    cnnNet = netInitialization(sys.argv[1], sys.argv[2])

    predictAndVisualize(cnnNet, sys.argv[3])
