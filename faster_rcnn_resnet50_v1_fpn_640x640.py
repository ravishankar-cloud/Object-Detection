%%capture
%pip install tensorflow_io sagemaker -U

import os
import sagemaker
from sagemaker.estimator import Estimator
from framework import CustomFramework

role = sagemaker.get_execution_role()
print(role)

# The train and val paths below are public S3 buckets created by Udacity for this project
inputs = {'train': 's3://cd2688-object-detection-tf2/train/', 
        'val': 's3://cd2688-object-detection-tf2/val/'} 

# Insert path of a folder in your personal S3 bucket to store tensorboard logs.
tensorboard_s3_prefix = 's3://object-detection-ravi/logs/'

%%bash

# clone the repo and get the scripts
git clone https://github.com/tensorflow/models.git docker/models

# get model_main and exporter_main files from TF2 Object Detection GitHub repository
cp docker/models/research/object_detection/exporter_main_v2.py source_dir 
cp docker/models/research/object_detection/model_main_tf2.py source_dir

# build and push the docker image. This code can be commented after being ran once.
# This will take around 10 mins.
image_name = 'tf2-object-detection'
!sh ./docker/build_and_push.sh $image_name

# display the container name
with open (os.path.join('docker', 'ecr_image_fullname.txt'), 'r') as f:
    container = f.readlines()[0][:-1]

print(container)

%%bash
mkdir /tmp/checkpoint
mkdir source_dir/checkpoint
wget -O /tmp/faster_rcnn_resnet50_v1_640x640.tar.gz http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
tar -zxvf /tmp/faster_rcnn_resnet50_v1_640x640.tar.gz --strip-components 2 --directory source_dir/checkpoint faster_rcnn_resnet50_v1_640x640/checkpoint

tensorboard_output_config = sagemaker.debugger.TensorBoardOutputConfig(
    s3_output_path=tensorboard_s3_prefix,
    container_local_output_path='/opt/training/'
)

estimator = CustomFramework(
    role=role,
    image_uri=container,
    entry_point='run_training.sh',
    source_dir='source_dir/',
    hyperparameters={
        "model_dir":"/opt/training",        
        "pipeline_config_path": "faster_rcnn_resnet50_v1_fpn_640x640.config",
        "num_train_steps": "3000",    
        "sample_1_of_n_eval_examples": "1"
    },
    instance_count=1,
    instance_type='ml.m5.4xlarge',
    tensorboard_output_config=tensorboard_output_config,
    disable_profiler=True,
    base_job_name='tf2-object-detection'
)

estimator.fit(inputs)

# your writeup goes here. 
1) Got the train and validation data from public s3 bucket
 2) To train the model, i have built a docker container with all the dependencies required by the TF Object Detection API and pushed it to ECR
 3) Downloaded and Applied a pre trained model faster_rcnn_resnet50_v1_fpn_640x640
 4) Have changed the config file according to  faster_rcnn_resnet50_v1_fpn_640x640 and experiemented with horizontal flipping Augmentation techniqueus.
 5) Have fine tuned hyperparameters num_classes and fine-tuned various aspect_ratios and anchor_scale.
 6) Deployed the model with the artifact generated from trained job from Sagemaker
 7) Validation Loss and Training Loss converged around 55 epochs which is a good sign.
 8) We can fine tune various hyperparameters and converge the validation and training loss beofre 55 epochs.
 9) Achieved 31.4 mAP on COCO17 Val
