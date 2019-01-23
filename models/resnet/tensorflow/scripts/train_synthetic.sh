# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Specify hosts in the file `hosts`, ensure that the number of slots is equal to the number of GPUs on that host

# This script has been tested on DLAMI v17 and above

if [ -z "$1" ]
  then
    echo "Usage: "$0" <num_gpus>"
    exit 1
  else
    gpus=$1
fi

source activate tf-py3

echo "Launching training job with synthetic data using $gpus GPUs"
set -ex

if [ "$gpus" -ge 128 ]; then LARC_AND_SCALING=" --use_larc --loss_scale 256." ; else LARC_AND_SCALING=""; fi

# p3 instances have larger GPU memory, so a higher batch size can be used
GPU_MEM=`nvidia-smi --query-gpu=memory.total --format=csv,noheader -i 0 | awk '{print $1}'`
if [ $GPU_MEM -gt 15000 ] ; then BATCH_SIZE=256; else BATCH_SIZE=128; fi

# Training
mpirun -np $gpus -hostfile ~/hosts -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	-x TF_CPP_MIN_LOG_LEVEL=0 \
	python -W ignore train_imagenet_resnet_hvd.py \
	--synthetic -b $BATCH_SIZE --num_epochs 5 --clear_log $LARC_AND_SCALING
