#!/bin/bash
caffe_savepredictions_h5.py ${1} ${2} --model_def=${3} --snapshot=${4} --gpu --layer=${5} --show_result --hdf5_format=my00
