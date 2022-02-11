#!/bin/bash

git clone https://github.com/tensorflow/models.git

cd models/research
sudo apt update
sudo apt install -y protobuf-compiler

echo $(protoc --version)

protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .