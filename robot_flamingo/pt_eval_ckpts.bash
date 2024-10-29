#!/bin/bash

export EVALUTION_ROOT=$(pwd)

# !!! Set for your own path
# calvin_dataset_path='YOUR_PATH/calvin/dataset/task_D_D'
calvin_dataset_path='/mnt/bn/yueyang/archive/calvin/dataset/task_D_D'
# calvin_conf_path
# calvin_conf_path="YOUR_PATH/calvin/calvin_models/conf"
calvin_conf_path="/mnt/bn/yueyang/archive/calvin/calvin_models/conf"

use_gripper=1
use_state=0

evaluate_from_checkpoint=$1
log_file=$2
window_size=$3
node_num=$4
amp=$5
exit_ratio=${6}
num_seq=${7}
max_layer=${8}
diverse_inst=${9}
precision=${10}

export MESA_GL_VERSION_OVERRIDE=4.1
echo logging to ${log_file}

script=eval_calvin.py
echo "EVAL IN LONG HORIZON MODE"

PORT=$((RANDOM % 16383 + 49152))

torchrun --nnodes=1 --nproc_per_node=${node_num}  --master_port=$PORT robot_flamingo/eval/$script \
    --precision ${precision} \
    --use_gripper \
    --diverse_inst ${diverse_inst} \
    --window_size ${window_size} \
    --run_name DeeR \
    --calvin_dataset ${calvin_dataset_path} \
    --cross_attn_every_n_layers 4 \
    --evaluate_from_checkpoint ${evaluate_from_checkpoint} \
    --calvin_conf_path ${calvin_conf_path} \
    --amp ${amp} \
    --exit_ratio ${exit_ratio} \
    --max_layer ${max_layer} \
    --num_seq ${num_seq} \
    --validation_set \
    --workers 1 > ${log_file} 2>&1
