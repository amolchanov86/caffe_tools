#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import argparse
import glob
import time
import matplotlib.pyplot as plt
from skimage import io

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
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'caffe/imagenet/ilsvrc_2012_mean.npy'),
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
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--ext",
        default='png',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    parser.add_argument(
        "--labels",
        default='val.txt',
        help='file with true labels')
    parser.add_argument(
        "--mis_file",
        default='misclassify.txt',
        help='output file with misclassifications')
    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    # Make classifier.
#    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
#            image_dims=image_dims, gpu=args.gpu, mean=mean,
#            input_scale=args.input_scale, raw_scale=args.raw_scale,
#            channel_swap=channel_swap)
    
    print 'Parameters:'
    print args.model_def, args.pretrained_model, image_dims, args.gpu, mean,args.input_scale, args.raw_scale,channel_swap
    
    classifier = caffe.Classifier(args.model_def, args.pretrained_model, image_dims, mean, args.input_scale, args.raw_scale, channel_swap)

    if args.gpu:
        print 'GPU mode'
        
    #Open misclassification file
    mis_file = open(args.mis_file, 'w')

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    args.input_file = os.path.expanduser(args.input_file)
    if args.input_file.endswith('npy'):
        inputs = np.load(args.input_file)

    elif os.path.isdir(args.input_file):
        try:
            data = list(map(str.split, open( args.labels )))
        except:
            print 'ERROR : Can not open the file with labels = ', args.labels
        pathToSaveMissed = args.input_file + os.path.sep + "missed" + os.path.sep
        pathToSaveFP = args.input_file + os.path.sep + "falseNegative" + os.path.sep
        
        for im_f in data:
            start = time.time()
            inputs = [caffe.io.load_image(args.input_file + im_f[0])]
            predictions = classifier.predict(inputs, not args.center_only)
            
            # Misclassified
            if((predictions[0].argmax() != int(im_f[1]))):
                print "Missclassified : True Class : " + im_f[1] + " , Predicted class : " + str(predictions[0].argmax()) + " , ImgName: " + im_f[0]
                if not os.path.exists(pathToSaveMissed):
                    os.mkdir(pathToSaveMissed)
                io.imsave( pathToSaveMissed + im_f[0], inputs)
                mis_file.write(im_f[0] + ' ' + im_f[1] + ' ' + str(predictions[0].argmax()) + np.array_str(predictions[0])[1:-1] + '\n')
                
                
            #False Negative 
            if((int(im_f[1]) == 0) and (predictions[0].argmax() != int(im_f[1]))):
                if not os.path.exists(pathToSaveFP):
                    os.mkdir(pathToSaveFP)
                io.imsave(pathToSaveFP + im_f[0], inputs)
                print "False negative : Actual Class : " + im_f[1] + " , predicted class : " + str(predictions[0].argmax()) + " , ImgName: " + im_f[0]
        return
    else:
        inputs = [caffe.io.load_image(args.input_file)]

    print "Classifying %d inputs." % len(inputs)

    # Classify.
    start = time.time()
    predictions = classifier.predict(inputs, not args.center_only)
    print "Done in %.2f s." % (time.time() - start)

    print 'Prediction Shape : ', predictions[0].shape
    plt.plot(predictions[0])
    print 'predicted class :', predictions[0].argmax()
    # Save
    np.save(args.output_file, predictions)


if __name__ == '__main__':
    main(sys.argv)


