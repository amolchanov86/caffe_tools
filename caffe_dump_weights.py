#!/usr/bin/env python

import caffe
import sys
import argparse
import h5py
import os
import sys

## NOTES:
# Convolutional filter dimensions order:
# TF :          H xx W xx IN_ch xx OUT_ch
# Caffe:        OUT_ch xx IN_ch xx H xx W


def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "model",
        help="Prototxt with model"
    )
    parser.add_argument(
        "snapshot",
        help="caffemodel with weights"
    )
    parser.add_argument(
        "-h5","--h5_filename",
        help="output hdf5 filename"
    )

    args = parser.parse_args()

    net = caffe.Net(args.model,
                    args.snapshot,
                    caffe.TEST)


    # Creating HDF5 dataset
    if args.h5_filename is None:
        name =  os.path.basename( os.path.splitext(args.model)[0] )
        output_file = name + "_weights.h5"
    else:
        output_file = args.h5_filename


    h5datafile = h5py.File(output_file, "w")

    # Getting all parameter names
    params = net.params.keys()
    # fc_params = {name: (weights, biases)}
    # fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}


    ## Crawling all parameters and dumping weitghts
    for param in params:
        print 'Dimensions order (W) = (out_ch, in_ch, h, w) '
        if len(net.params[param]) == 2:
            print 'Param = ', param, 'W = ', net.params[param][0].data.shape, ' B = ', net.params[param][1].data.shape
        else:
            print 'Param = ', param, 'W = ', net.params[param][0].data.shape

        # Saving in hdf5
        blobs_num = len(net.params[param])
        for blob in range(0, blobs_num):
            h5datafile[param + ('/%d' % blob)] = net.params[param][blob].data

    ## Cleaning
    h5datafile.close()

if __name__ == '__main__':
    main(sys.argv)