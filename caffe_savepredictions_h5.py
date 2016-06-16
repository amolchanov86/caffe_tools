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

# from google.protobuf import text_format
# from caffe.draw import get_pydot_graph
# from caffe.proto import caffe_pb2
# from IPython.display import display, Image

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input file"
    )
    parser.add_argument(
        "--output_file",
        default='',
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
    parser.add_argument(
       "--mean_px",
       default='122,122,122',
       help="Mean pixel: BGR"
    )
    parser.add_argument(
       "--temperatures",
       default='1',
       help="Temperatures that will be applied to activations"
    )
    args = parser.parse_args()

    # Need it for temperatures
    # def softmax(x):
    #     """Compute softmax values for each sets of scores in x."""
    #     return np.exp(x) / np.sum(np.exp(x), axis=0)

    #Softmax for images
    def softmax(x):
        exp_mx = np.exp(x)
        sum_mx = np.sum(exp_mx, axis=2)
        # print 'sum_mx shape = ', sum_mx.shape
        res_mx = np.zeros(exp_mx.shape)
        labels_num = res_mx.shape[2]
        for ch_i in range(0,labels_num):
            res_mx[:,:, ch_i] = np.divide(exp_mx[:,:, ch_i], sum_mx)
        return res_mx
        # print 'res_shape', res_mx.shape



    # Parameters
    lbl_img_scale = 50
    act_img_scale = 50

    if args.mean_px:
        mean_bgr = [float(s) for s in args.mean_px.split(',')]
        mean_b = mean_bgr[0]
        mean_g = mean_bgr[1]
        mean_r = mean_bgr[2]
    else:
        mean_b = 122
        mean_g = 122
        mean_r = 122
        # For GoogleNet
        # mean_b = 104.00699
        # mean_g = 116.66877
        # mean_r = 122.67892

    #Temperatures
    if args.temperatures:
        temperature = [float(s) for s in args.temperatures.split(',')]
    else:
        temperature = [1.0]

    mean, channel_swap = None, None
    if args.mean_file and len(args.mean_file):
        mean = np.load(args.mean_file)

    preproc = 0

    print 'Input file = ', args.input_file

    # Loading validation file
    val_set_file = h5py.File(args.input_file, 'r')
    print 'HDF5 file open'
    if args.hdf5_format == 'caffe':
        print 'Caffe input file format'
        data_var = val_set_file['/data']
        label_var = val_set_file['/label']
    elif args.hdf5_format == 'my00':
        print 'MY00 input file format'
        data_var = val_set_file['/feat/img']
        label_var = val_set_file['/label/img']
        preproc = 1
    else:
        print 'ERROR: unknown HDF5 file format !!!'
        val_set_file.close()
        sys.exit()

    # Not good actually since I am loading all images in the memory
    # change it later
    inputs = data_var[:]
    labels = label_var[:]

    print 'Input file =',  args.input_file , 'Labels shape =', labels.shape, '  Data shape =', inputs.shape
    print 'Mean pixel (BGR) = ', mean_b, mean_g, mean_r
    print 'Temperatures = ', temperature



    channels_num = inputs.shape[1]
    samples_num = inputs.shape[0]

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
    # print 'Mean shape = ', mean_img.shape
    # print mean_img[0, :]


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
    activations = net.blobs[layer_name].data

    save_data = False
    if args.output_file != '':
        # Opening HDF5 file
        h5datafile = h5py.File(args.output_file, "w")

        save_data = True
        act_shape = [samples_num, activations.shape[2], activations.shape[3], activations.shape[1]];
        feat_dset  = h5datafile.create_dataset("/feat/img",    inputs.shape, dtype='uint8')
        label_dset = h5datafile.create_dataset("/label/label", labels.shape, dtype='uint8')
        logit_dset = h5datafile.create_dataset("/label/pred_logit", act_shape, dtype='float32')

        temp_dsets = []
        for temp_i in temperature:
            temp_dset_cur = h5datafile.create_dataset("/label/pred_t%3.1f" % temp_i, act_shape, dtype='float32')
            temp_dsets.append(temp_dset_cur)

    # Fixing the start time
    start_time = time.time()

    print 'input shape  =', inputs.shape
    print 'labels shape = ', labels.shape
    print 'activations shape = ', activations.shape

    print 'Sample-by-sample prediction ...'
    for img_indx in range(inputs.shape[0]):
        img = np.expand_dims( inputs[img_indx, :], 0)
        img = img.astype(np.float32)
        img = img - mean_img
        if preproc == 1:
            img = np.swapaxes(img, 3, 2)
            img = np.swapaxes(img, 2, 1)

        label_img = np.squeeze(labels[img_indx])

        print '----------------------------------'

        #Forward pass
        net.blobs['data'].data[...] = img
        net.forward()
        # out = net.forward_all(data=img)

        #Getting data
        activations = net.blobs[layer_name].data

        #Get rid of the first singleton dimension
        activations = np.squeeze(activations)

        #Getting label
        pred_lbl_img = np.argmax(activations, 0)

        print 'Img# =', img_indx

        #Scaling for visualization
        pred_img_scaled = act_img_scale * pred_lbl_img
        lbl_image_scaled = lbl_img_scale * label_img

        if args.show_result:
            cv2.imshow('input_img', inputs[img_indx, :])
            cv2.imshow('label_img', lbl_image_scaled)
            cv2.imshow("predictions for %s" % layer_name, pred_img_scaled.astype(np.uint8))

            cv2.waitKey(0)

        activations_swaped = np.swapaxes(activations, 0, 1)
        activations_swaped = np.swapaxes(activations_swaped, 1, 2)

        if save_data:
            feat_dset[img_indx, :]  = inputs[img_indx, :]
            label_dset[img_indx, :] = labels[img_indx, :]
            logit_dset[img_indx, :] = activations_swaped

            temp_i = -1
            for temp_cur in temperature:
                temp_i += 1
                soft_activations = softmax(activations_swaped / temp_cur)
                print 'px_dist(1,1): T = ', temp_cur, soft_activations[1,1,:]
                temp_dsets[temp_i][img_indx, :] = soft_activations


    val_set_file.close()

if __name__ == '__main__':
    main(sys.argv)


