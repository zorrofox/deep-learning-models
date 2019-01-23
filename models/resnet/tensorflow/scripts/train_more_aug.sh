# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Specify hosts in the file `hosts`, ensure that the number of slots is equal to the number of GPUs on that host

# Use this script when training with large number of GPUs (128, 256, etc). It uses more augmentations than train.sh, and also uses layer wise adaptive rate control (LARC) to help with convergence at large batch sizes.

# This script has been tested on DLAMI v17 and above

if [ -z "$1" ]
  then
    echo "Usage: "$0" <num_gpus>"
    exit 1
  else
    gpus=$1
fi

source activate tf-py3

echo "Launching training job using $gpus GPUs"
set -ex

NUM_GPUS_MASTER=`nvidia-smi -L | wc -l`

# p3 instances have larger GPU memory, so a higher batch size can be used
GPU_MEM=`nvidia-smi --query-gpu=memory.total --format=csv,noheader -i 0 | awk '{print $1}'`
if [ $GPU_MEM -gt 15000 ] ; then BATCH_SIZE=256; else BATCH_SIZE=128; fi

# Training
# This script is for training with large number of GPUs (large batch sizes). 
# You can for instance just replace the number of GPUs to 128 with the same script.
mpirun -np $gpus -hostfile ~/hosts -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	-x TF_CPP_MIN_LOG_LEVEL=0 \
	python -W ignore train_imagenet_resnet_hvd.py \
	--data_dir ~/data/tf-imagenet/ --num_epochs 90 --increased_aug -b $BATCH_SIZE \
	--mom 0.977 --wdecay 0.0005 --loss_scale 256. --use_larc \
	--lr_decay_mode linear_cosine --warmup_epochs 5 --clear_log

# Evaluation
# Using only gpus on master node for evaluation as we saved checkpoints only on master node
# pass num_gpus it was trained on to print the epoch numbers correctly
mpirun -np $NUM_GPUS_MASTER -mca plm_rsh_no_tree_spawn 1 \
	-bind-to socket -map-by slot \
	-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
	-x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
	-x TF_CPP_MIN_LOG_LEVEL=0 \
	python -W ignore train_imagenet_resnet_hvd.py \
	--data_dir ~/data/tf-imagenet/ --num_epochs 90 -b $BATCH_SIZE \
	--eval --num_gpus $gpus
