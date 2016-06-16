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


## Functions applies bypixel softmax
def softmax(x):
    exp_mx = np.exp(x)
    sum_mx = np.sum(exp_mx, axis=2)
    # print 'sum_mx shape = ', sum_mx.shape
    res_mx = np.zeros(exp_mx.shape)
    labels_num = res_mx.shape[2]
    for ch_i in range(0, labels_num):
        res_mx[:, :, ch_i] = np.divide(exp_mx[:, :, ch_i], sum_mx)
    return res_mx

## Function defines default channgel swap (reverse channel order)
# @param channels_num Number of channels
def default_chswap(channels_num):
    print 'Applying default channel swap ...'
    channel_swap = []
    for ch_i in range(channels_num - 1, -1, -1):
        channel_swap.append(ch_i)
    return channel_swap

## Function to save predictions
#@param mean - could be 1 of 3 options: string = mean file to load, numpy array = mean image, list/tuple = mean pixel in BGR order
def savepredicitons_h5(input_file,
                       output_file,
                       model_def,
                       snapshot,
                       temperatures=[1.0],
                       mean = [104,116,122],
                       layer_name="upsample2",
                       hdf5_format="my00",
                       gpu=True,
                       show_results=False):

    # Parameters
    lbl_img_scale = 50
    act_img_scale = 50

    preproc = 0

    print 'Input file = ', input_file

    # Loading validation file
    val_set_file = h5py.File(input_file, 'r')
    print 'HDF5 file open'
    if hdf5_format == 'caffe':
        print 'Caffe input file format'
        data_var = val_set_file['/data']
        label_var = val_set_file['/label']
    elif hdf5_format == 'my00':
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

    if type(mean) == np.ndarray:
        mean_img = mean
    elif type(mean) == list or type(mean) == tuple:
        image_dims = [inputs.shape[2], inputs.shape[3]]
        mean_img = np.ones(inputs.shape[1:])
        mean_img[0, :] = mean[0]
        mean_img[1, :] = mean[1]
        mean_img[2, :] = mean[2]
    elif type(mean) == str:
        mean_img = np.load(mean)
    else:
        image_dims = [inputs.shape[2], inputs.shape[3]]
        mean_img = np.ones(inputs.shape[1:])
        mean_img[0, :] = 122
        mean_img[1, :] = 122
        mean_img[2, :] = 122


    print 'Input file =', input_file, 'Labels shape =', labels.shape, '  Data shape =', inputs.shape
    print 'Temperatures = ', temperatures

    channels_num = inputs.shape[1]
    samples_num = inputs.shape[0]

    # Printing parameters
    print 'Image dimensions =', image_dims
    print '-----------------------------------------------------------------------------'

    if gpu:
        print 'GPU mode'
        caffe.set_mode_gpu()
    else:
        print 'CPU mode'
        caffe.set_mode_cpu()

    # Loading the NET
    # classifier = caffe.Classifier(args.model_def, args.snapshot)
    net = caffe.Net(model_def, snapshot, caffe.TEST)
    activations = net.blobs[layer_name].data

    save_data = False
    if output_file != '':
        # Opening HDF5 file
        h5datafile = h5py.File(output_file, "w")

        save_data = True
        act_shape = [samples_num, activations.shape[2], activations.shape[3], activations.shape[1]];
        feat_dset = h5datafile.create_dataset("/feat/img", inputs.shape, dtype='uint8')
        label_dset = h5datafile.create_dataset("/label/label", labels.shape, dtype='uint8')
        logit_dset = h5datafile.create_dataset("/label/pred_logit", act_shape, dtype='float32')

        temp_dsets = []
        for temp_i in temperatures:
            temp_dset_cur = h5datafile.create_dataset("/label/pred_t%3.1f" % temp_i, act_shape, dtype='float32')
            temp_dsets.append(temp_dset_cur)

    # Fixing the start time
    start_time = time.time()

    print 'input shape  =', inputs.shape
    print 'labels shape = ', labels.shape
    print 'activations shape = ', activations.shape

    print 'Sample-by-sample prediction ...'
    for img_indx in range(inputs.shape[0]):
        img = np.expand_dims(inputs[img_indx, :], 0)
        img = img.astype(np.float32)
        img = img - mean_img
        if preproc == 1:
            img = np.swapaxes(img, 3, 2)
            img = np.swapaxes(img, 2, 1)

        label_img = np.squeeze(labels[img_indx])

        print '----------------------------------'

        # Forward pass
        net.blobs['data'].data[...] = img
        net.forward()
        # out = net.forward_all(data=img)

        # Getting data
        activations = net.blobs[layer_name].data

        # Get rid of the first singleton dimension
        activations = np.squeeze(activations)

        # Getting label
        pred_lbl_img = np.argmax(activations, 0)

        print 'Img# =', img_indx

        # Scaling for visualization
        pred_img_scaled = act_img_scale * pred_lbl_img
        lbl_image_scaled = lbl_img_scale * label_img

        if show_results:
            cv2.imshow('input_img', inputs[img_indx, :])
            cv2.imshow('label_img', lbl_image_scaled)
            cv2.imshow("predictions for %s" % layer_name, pred_img_scaled.astype(np.uint8))

            cv2.waitKey(0)

        activations_swaped = np.swapaxes(activations, 0, 1)
        activations_swaped = np.swapaxes(activations_swaped, 1, 2)

        if save_data:
            feat_dset[img_indx, :] = inputs[img_indx, :]
            label_dset[img_indx, :] = labels[img_indx, :]
            logit_dset[img_indx, :] = activations_swaped

            temp_i = -1
            for temp_cur in temperatures:
                temp_i += 1
                soft_activations = softmax(activations_swaped / temp_cur)
                print 'px_dist(1,1): T = ', temp_cur, soft_activations[1, 1, :]
                temp_dsets[temp_i][img_indx, :] = soft_activations

    val_set_file.close()

    end_time = time.time()

    print '-------------------------------------------'
    print 'Processing time = ', end_time - start_time


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input file"
    )
    # Optional arguments.
    parser.add_argument(
        "model_def",
        help="Model definition file."
    )
    parser.add_argument(
        "snapshot",
        help="Trained model weights file."
    )
    parser.add_argument(
        "--output_file",
        default='',
        help="Output npy filename."
    )
    parser.add_argument(
        "--mean_file",
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
       "--mean_px",
       default='122,122,122',
       help="Mean pixel: BGR. Ignored if --mean_file is provided"
    )
    parser.add_argument(
       "--layer",
       default='prob',
       help="Layer to save activations from"
    )
    parser.add_argument(
        "--hdf5_format",
        default='my00',
        help="Which type of data storing we have: caffe (/data, /label), my00 (/feat/img, /label/img)"
    )
    parser.add_argument(
       "--temperatures",
       default='1',
       help="Temperatures that will be applied to activations"
    )
    parser.add_argument(
        "--show_result",
        action='store_true',
        help="Show resulting images"
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )

    args = parser.parse_args()
    print "Parameters: ", args
    print '#########################################################'

    #Temperatures
    if args.temperatures:
        temperatures = [float(s) for s in args.temperatures.split(',')]
    else:
        temperatures = [1.0]

    #Mean value
    if (args.mean_file is not None) and (args.mean_file != ''):
        mean = args.mean_file
    else:
        mean = args.mean_px


    savepredicitons_h5(args.input_file,
                       args.output_file,
                       args.model_def,
                       args.snapshot,
                       temperatures=temperatures,
                       mean=[104, 116, 122],
                       layer_name=args.layer,
                       hdf5_format=args.hdf5_format,
                       gpu=args.gpu,
                       show_results=args.show_result)



if __name__ == '__main__':
    main(sys.argv)


