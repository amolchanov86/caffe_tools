#!/bin/bash
caffe_savepredictions_h5.py ${1} \
	--model_def=${2} \
	--snapshot=${3} \
	--output_file=${4} \
	--layer=${5} \
	--hdf5_format=my00 \
	--mean_px=104,116,122
	--show_result \
	--gpu
