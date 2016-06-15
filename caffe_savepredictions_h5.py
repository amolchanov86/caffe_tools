#!/usr/bin/env python
"""
The scripti is meant to save activations
"""
import numpy as np
import os
import sys
import argparse
import time
import matplotlib.pyplot as plt
from skimage import io
import h5py
import cv2

import caffe

from google.protobuf import text_format
from caffe.draw import get_pydot_graph
from caffe.proto import caffe_pb2
from IPython.display import display, Image

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
        "--snapshot",
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
        default=1.0,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=1.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    # parser.add_argument(
    #    "--channel_swap",
    #    default='',
    #    help="Order to permute input channels. The default converts " +
    #         "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    # )
    parser.add_argument(
       "--layer",
       default='prob',
       help="Layer to save activations from"
    )
    parser.add_argument(
        "--show_result",
        action='store_true',
        help="Show resulting images"
    )
    parser.add_argument(
        "--hdf5_format",
        default='caffe',
        help="Which type of data storing we have: caffe (/data, /label), my00 (/feat/img, /label/img)"
    )
    args = parser.parse_args()


    # Parameters
    lbl_img_scale = 50
    act_img_scale = 50
    mean_b = 104.00699
    mean_g = 116.66877
    mean_r = 122.67892
    # mean_b = 122.00699
    # mean_g = 122.66877
    # mean_r = 122.67892

    mean, channel_swap = None, None
    if args.mean_file and len(args.mean_file):
        mean = np.load(args.mean_file)

    # Loading validation file
    val_set_file = h5py.File(args.input_file, 'r')

    preproc = 0

    if args.hdf5_format == 'caffe':
        data_var = val_set_file['data']
        label_var = val_set_file['label']
    elif args.hdf5_format == 'my00':
        data_var = val_set_file['/feat/img']
        label_var = val_set_file['/label/img']
        preproc = 1
    else:
        print 'ERROR: unknown HDF5 file format !!!'
        val_set_file.close()
        sys.exit()

    inputs = data_var[:]
    labels = label_var[:]
    # inputs = inputs.astype(np.float32)

    # inputs = np.zeros(data_var.shape)
    # labels = np.zeros(data_var.shape)

    # Not good actually since I am loading all images in the memory
    # change it later

    
    print 'Input file =',  args.input_file , 'Labels shape =' , labels.shape  , '  Data shape =', inputs.shape    

    channels_num = inputs.shape[1]

    # channel_swap = []
    # def default_chswap(channels_num):
    #     print 'Applying default channel swap ...'
    #     for ch_i in range(channels_num - 1, -1, -1):
    #         channel_swap.append(ch_i)

    # Creating channel swap
    # if args.channel_swap:
    #     if args.channel_swap == '':
    #         channel_swap = default_chswap(channels_num)
    #     else:
    #         channel_swap = [int(s) for s in args.channel_swap.split(',')]
    # else:
    #     channel_swap = default_chswap(channels_num)

    image_dims = [inputs.shape[2], inputs.shape[3]]
    mean_img = np.ones(inputs.shape[1:])
    mean_img[0, :] = mean_b
    mean_img[1, :] = mean_g
    mean_img[2, :] = mean_r
    print 'Mean shape = ', mean_img.shape
    print mean_img[0, :]


    # Exctracting layer name
    layer_name = args.layer


    # Printing parameters
    # print 'Channel swap = ', channel_swap
    print 'Image dimensions =', image_dims

    print 'Parameters:'
    print args.model_def, args.snapshot, image_dims, args.gpu, mean, args.input_scale, args.raw_scale

    print '-----------------------------------------------------------------------------'

    if args.gpu:
        print 'GPU mode'
        caffe.set_mode_gpu()
    else:
        print 'CPU mode'
        caffe.set_mode_cpu()

    # Loading the NET
    # classifier = caffe.Classifier(args.model_def, args.snapshot)
    net = caffe.Net(args.model_def, args.snapshot, caffe.TEST)

    # print net.layer_names

    # Fixing the start time
    start_time = time.time()

    print 'Inputs shape =', inputs.shape

    print 'Sample-by-sample prediction ...'
    for img_indx in range(inputs.shape[0]):
    # for img_indx in range(2):

        img = np.expand_dims( inputs[img_indx, :], 0)
        img = img.astype(np.float32)
        img = img - mean_img
        # img = img / 255
        if preproc == 1:
            img = np.swapaxes(img, 3, 2)
            img = np.swapaxes(img, 2, 1)

        label_img = np.squeeze(labels[img_indx])


        print 'Iter=', img_indx, \
            'Input shape  =', img.shape

        net.blobs['data'].data[...] = img
        net.forward()
        # out = net.forward_all(data=img)

        # activations = out[layer_name][0]
        activations = net.blobs[layer_name].data
        activations = np.squeeze(activations)
        print 'activations shape = ', activations.shape
        act_img = np.argmax(activations, 0)
        print 'prediction img shape =', act_img.shape
        act_scaled = act_img_scale * act_img
        lbl_image_scaled = lbl_img_scale * label_img
        print 'Pred img shape = ', act_scaled.shape
        print 'Label shape = ', lbl_image_scaled.shape

        if args.show_result:
            cv2.imshow('data_img', inputs[img_indx, :])
            cv2.imshow('lbl_img', lbl_image_scaled)
            cv2.imshow("activations for %s" % layer_name, act_scaled.astype(np.uint8))

            # imgplot = plt.imshow(activations[0,:])
            # mgplot = plt.imshow(img[0,0,:])


            # fig = plt.figure(figsize=(10, 10))
            # ax = fig.add_subplot(111)
            #
            # cax = ax.matshow(activations[0,:, :], interpolation='none')
            # fig.colorbar(cax, orientation="horizontal")

            cv2.waitKey(0)

    val_set_file.close()

if __name__ == '__main__':
    main(sys.argv)


