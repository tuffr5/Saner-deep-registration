#!/bin/bash

# this example uses a single node (`NUM_NODES=1`) w/ 3 GPUs (`NUM_GPUS_PER_NODE=4`)
export NUM_NODES=1
export NUM_GPUS_PER_NODE=3
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

# launch your script w/ `torch.distributed.launch`
python -m torch.distributed.launch \
	--nproc_per_node=$NUM_GPUS_PER_NODE \
	--nnodes=$NUM_NODES \
	--node_rank $NODE_RANK \
	--master_port 44444 \
	pnp_admms_image_registration_ddp.py \
