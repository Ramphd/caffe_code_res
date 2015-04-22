#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

../build/tools/compute_image_mean ~/caffe_150323/CarStyle_test/carstyle_train_lmdb \
  ~/caffe_150323/CarStyle_test/car_mean.binaryproto

echo "Done."
