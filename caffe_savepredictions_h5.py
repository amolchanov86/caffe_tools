#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import argparse
import time
import matplotlib.pyplot as plt
from skimage import io
import h5py

import caffe


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "output_file",
        help="Output npy filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--mean_file",
        default='',
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=1.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
       "--channel_swap",
       default='',
       help="Order to permute input channels. The default converts " +
            "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
       "--layer",
       default='prob',
       help="layer to save activations from"
    )
    args = parser.parse_args()


    #Other options
    batch_prediction = True

    mean, channel_swap = None, None
    if args.mean_file and len(args.mean_file):
        mean = np.load(args.mean_file)


    # Loading validation file
    val_set_file = h5py.File(args.input_file, 'r')

    data_var = val_set_file['data']
    label_var = val_set_file['label']

    # inputs = np.zeros(data_var.shape)
    # labels = np.zeros(data_var.shape)
    
    inputs = data_var[:]
    labels = label_var[:]    

    
    print 'Input file =',  args.input_file , 'Labels shape =' , labels.shape  , '  Data shape =', inputs.shape    

    image_dims = [inputs.shape[2], inputs.shape[3]]
    channels_num = inputs.shape[1]

    def default_chswap(channels_num):
        for ch_i in range(channels_num - 1, -1, -1):
            channel_swap.append(ch_i)

    # Creating channel swap
    if args.channel_swap:
        if args.channel_swap == '':
            channel_swap = default_chswap(channels_num)
        else:
            channel_swap = [int(s) for s in args.channel_swap.split(',')]
    else:
        channel_swap = default_chswap(channels_num)

    # Exctracting layer name
    layer_name = args.layer


    # Printing parameters
    print 'Channel swap = ', channel_swap    
    print 'Image dimensions =', image_dims

    print 'Parameters:'
    print args.model_def, args.pretrained_model, image_dims, args.gpu, mean, args.input_scale, args.raw_scale, channel_swap 

    print '-----------------------------------------------------------------------------'

    if args.gpu:
        print 'GPU mode'
        caffe.set_mode_gpu()
    else:
        print 'CPU mode'
        caffe.set_mode_cpu()

    # Loading the NET
    # classifier = caffe.Classifier(args.model_def, args.pretrained_model)
    net = caffe.Net(args.model_def, args.pretrained_model, caffe.TEST)

    # Fixing the start time
    start = time.time()


    #Rearranging axis
    inputs = np.swapaxes(inputs, 1,2)
    inputs = np.swapaxes(inputs, 2,3)
    inputs = inputs.astype(np.float32)
    print 'Inputs shape =', inputs.shape

    miscl_count = 0

    if batch_prediction:
        print 'Batch prediction ...'
        data_in = []
        for img_indx in range(inputs.shape[0]):
            data_in.append(inputs[img_indx,:])
        print 'Samples to predict = ', len(data_in)
        predictions = classifier.predict( data_in, False )

        for img_indx in range(inputs.shape[0]):
            print "True Class : ", labels[img_indx], " , Predicted class : ",  predictions[img_indx].argmax(), ' distribution = ', predictions[img_indx]
            if labels[img_indx] != predictions[img_indx].argmax():
                miscl_count = miscl_count + 1

    else:
        print 'Sample-by-sample prediction ...'
        for img_indx in range(inputs.shape[0]):
#                print 'Shape =', inputs[img_indx,:].shape            
            predictions = classifier.predict( [inputs[img_indx,:]], False )
            #print 'Pred shape = ', predictions.shape
            print "True Class : ", labels[img_indx], " , Predicted class : ",  predictions[0].argmax(), ' distribution = ', predictions[0]
            if labels[img_indx] != predictions[0].argmax():
                miscl_count = miscl_count + 1

    print 'Accuracy = ', 1. - float(miscl_count) / float(inputs.shape[0]), ' Total = ', float(inputs.shape[0])

    val_set_file.close()

if __name__ == '__main__':
    main(sys.argv)


