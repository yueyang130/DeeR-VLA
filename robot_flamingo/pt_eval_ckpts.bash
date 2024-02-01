#!/bin/bash

export EVALUTION_ROOT=$(pwd)

# !!! Set for your own path
# calvin_dataset_path='calvin_data/task_ABCD_D'
calvin_dataset_path='/mnt/bn/yueyang/archive/calvin/dataset/task_ABCD_D'
# calvin_conf_path
calvin_conf_path="/mnt/bn/yueyang/archive/calvin/calvin_models/conf"
# language model path
lm_path=''
# tokenizer path
tokenizer_path=''

evaluate_from_checkpoint=$1
log_file=$2
use_gripper=$3
use_state=$4
fusion_mode=$5
window_size=$6
node_num=$7
single_step=$8
amp=$9
# eval_exit_mode=$10
eval_exit_mode=${10}
export MESA_GL_VERSION_OVERRIDE=4.1
echo logging to ${log_file}

if [ $single_step = true ]; then
    script=eval_calvin_single_step.py
    echo "EVAL IN SINGLE STEP MODE"
else
    script=eval_calvin.py
    echo "EVAL IN LONG HORIZON MODE"
fi

PORT=$((RANDOM % 16383 + 49152))

if [ ${use_gripper} -eq 1 ] && [ ${use_state} -eq 1 ]
then
torchrun --nnodes=1 --nproc_per_node=${node_num}  --master_port=$PORT robot_flamingo/eval/$script \
    --precision fp32 \
    --use_gripper \
    --use_state \
    --window_size ${window_size} \
    --fusion_mode ${fusion_mode} \
    --run_name RobotFlamingoDBG \
    --calvin_dataset ${calvin_dataset_path} \
    --cross_attn_every_n_layers 4 \
    --evaluate_from_checkpoint ${evaluate_from_checkpoint} \
    --calvin_conf_path ${calvin_conf_path} \
    --amp ${amp} \
    --eval_exit_mode ${eval_exit_mode} \
    --workers 1 > ${log_file} 2>&1
fi

if [ ${use_gripper} -eq 1 ] && [ ${use_state} -eq 0 ]
then
torchrun --nnodes=1 --nproc_per_node=${node_num}  --master_port=$PORT robot_flamingo/eval/$script \
    --precision fp32 \
    --use_gripper \
    --window_size ${window_size} \
    --fusion_mode ${fusion_mode} \
    --run_name RobotFlamingoDBG \
    --calvin_dataset ${calvin_dataset_path} \
    --cross_attn_every_n_layers 4 \
    --evaluate_from_checkpoint ${evaluate_from_checkpoint} \
    --calvin_conf_path ${calvin_conf_path} \
    --amp ${amp} \
    --eval_exit_mode ${eval_exit_mode} \
    --workers 1 > ${log_file} 2>&1
fi

if [ ${use_gripper} -eq 0 ] && [ ${use_state} -eq 0 ]
then
torchrun --nnodes=1 --nproc_per_node=${node_num}  --master_port=$PORT robot_flamingo/eval/$script \
    --precision fp32 \
    --run_name RobotFlamingoDBG \
    --window_size ${window_size} \
    --fusion_mode ${fusion_mode} \
    --calvin_dataset ${calvin_dataset_path} \
    --cross_attn_every_n_layers 4 \
    --evaluate_from_checkpoint ${evaluate_from_checkpoint} \
    --calvin_conf_path ${calvin_conf_path} \
    --amp ${amp} \
    --eval_exit_mode ${eval_exit_mode} \
    --workers 1 > ${log_file} 2>&1
fi