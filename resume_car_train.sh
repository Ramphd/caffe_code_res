#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=/home/cari/caffe_150323/CarStyle_test/model_files/car_solver.prototxt \
    --snapshot=/home/cari/caffe_150323/CarStyle_test/trained_model/327_iter_400.solverstate
