#!/bin/bash

export EVALUTION_ROOT=$(pwd)

# !!! Set for your own path
# calvin_dataset_path='calvin_data/task_ABCD_D'
calvin_dataset_path='/mnt/bn/yueyang/archive/calvin/dataset/calvin_debug_dataset'
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
export MESA_GL_VERSION_OVERRIDE=4.1
echo logging to ${log_file}
node_num=2

if [ ${use_gripper} -eq 1 ] && [ ${use_state} -eq 1 ]
then
torchrun --nnodes=1 --nproc_per_node=${node_num}  --master_port=6066 robot_flamingo/eval/eval_calvin.py \
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
    --workers 1 > ${log_file} 2>&1
fi

if [ ${use_gripper} -eq 1 ] && [ ${use_state} -eq 0 ]
then
torchrun --nnodes=1 --nproc_per_node=${node_num}  --master_port=6099 robot_flamingo/eval/eval_calvin.py \
    --precision fp32 \
    --use_gripper \
    --window_size ${window_size} \
    --fusion_mode ${fusion_mode} \
    --run_name RobotFlamingoDBG \
    --calvin_dataset ${calvin_dataset_path} \
    --cross_attn_every_n_layers 4 \
    --evaluate_from_checkpoint ${evaluate_from_checkpoint} \
    --calvin_conf_path ${calvin_conf_path} \
    --workers 1 > ${log_file} 2>&1
fi

if [ ${use_gripper} -eq 0 ] && [ ${use_state} -eq 0 ]
then
torchrun --nnodes=1 --nproc_per_node=${node_num}  --master_port=6066 robot_flamingo/eval/eval_calvin.py \
    --precision fp32 \
    --run_name RobotFlamingoDBG \
    --window_size ${window_size} \
    --fusion_mode ${fusion_mode} \
    --calvin_dataset ${calvin_dataset_path} \
    --cross_attn_every_n_layers 4 \
    --evaluate_from_checkpoint ${evaluate_from_checkpoint} \
    --calvin_conf_path ${calvin_conf_path} \
    --workers 1 > ${log_file} 2>&1
fi