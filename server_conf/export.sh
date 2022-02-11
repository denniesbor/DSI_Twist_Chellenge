#!/bin/bash

cd /home/dkbor/Desktop/Twist_Chellenge/flask_/flasky/models/research/object_detection
PYTHONSCRIPT=exporter_main_v2.py
PIPELINECONFIG=/home/dkbor/Desktop/Twist_Chellenge/flask_/flasky/pipeline.config
OUTPUTDIR=/home/dkbor/Desktop/Twist_Chellenge/flask_/flasky/model
TRAINEDCHECKPOINTDIR=/home/dkbor/Desktop/Twist_Chellenge/flask_/flasky/data/checkpoints

python3 ${PYTHONSCRIPT}  \
	--input_type='image_tensor' \
	--pipeline_config_path=${PIPELINECONFIG}\
        --output_directory=${OUTPUTDIR} \
        --trained_checkpoint_dir=${TRAINEDCHECKPOINTDIR}