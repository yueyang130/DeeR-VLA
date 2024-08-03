#!/bin/bash

export EVALUTION_ROOT=$(pwd)

# !!! Set for your own path
# calvin_dataset_path='calvin_data/task_ABCD_D'
# calvin_dataset_path='/mnt/bn/yueyang/archive/calvin/dataset/task_ABCD_D'
calvin_dataset_path='/mnt/bn/yueyang/archive/calvin/dataset/task_D_D'
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
multi_execution=${11}
value_net_ckpt=${12}
exit_ratio=${13}
layerwise_exit_eval=${14}
value_type=${15}
num_seq=${16}
threshold_type=${17}
use_action_ensemble=${18}
exit_dist=${19}
max_layer=${20}
diverse_inst=${21}
precision=${22}

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
    --precision ${precision} \
    --use_gripper \
    --use_state \
    --diverse_inst ${diverse_inst} \
    --window_size ${window_size} \
    --fusion_mode ${fusion_mode} \
    --run_name RobotFlamingoDBG \
    --calvin_dataset ${calvin_dataset_path} \
    --cross_attn_every_n_layers 4 \
    --evaluate_from_checkpoint ${evaluate_from_checkpoint} \
    --calvin_conf_path ${calvin_conf_path} \
    --amp ${amp} \
    --eval_exit_mode ${eval_exit_mode} \
    --multi_execution ${multi_execution} \
    --value_net_ckpt ${value_net_ckpt} \
    --exit_ratio ${exit_ratio} \
    --workers 1 > ${log_file} 2>&1
fi

if [ ${use_gripper} -eq 1 ] && [ ${use_state} -eq 0 ]
then
torchrun --nnodes=1 --nproc_per_node=${node_num}  --master_port=$PORT robot_flamingo/eval/$script \
    --precision ${precision} \
    --use_gripper \
    --diverse_inst ${diverse_inst} \
    --window_size ${window_size} \
    --fusion_mode ${fusion_mode} \
    --run_name RobotFlamingoDBG \
    --calvin_dataset ${calvin_dataset_path} \
    --cross_attn_every_n_layers 4 \
    --evaluate_from_checkpoint ${evaluate_from_checkpoint} \
    --calvin_conf_path ${calvin_conf_path} \
    --amp ${amp} \
    --eval_exit_mode ${eval_exit_mode} \
    --multi_execution ${multi_execution} \
    --value_net_ckpt ${value_net_ckpt} \
    --exit_ratio ${exit_ratio} \
    --layerwise_exit_eval ${layerwise_exit_eval} \
    --value_type ${value_type} \
    --threshold_type ${threshold_type} --exit_dist ${exit_dist} --max_layer ${max_layer} \
    --use_action_ensemble ${use_action_ensemble} \
    --num_seq ${num_seq} \
    --validation_set \
    --workers 1 
    # --workers 1 > ${log_file} 2>&1
fi

if [ ${use_gripper} -eq 0 ] && [ ${use_state} -eq 0 ]
then
torchrun --nnodes=1 --nproc_per_node=${node_num}  --master_port=$PORT robot_flamingo/eval/$script \
    --precision ${precision} \
    --diverse_inst ${diverse_inst} \
    --run_name RobotFlamingoDBG \
    --window_size ${window_size} \
    --fusion_mode ${fusion_mode} \
    --calvin_dataset ${calvin_dataset_path} \
    --cross_attn_every_n_layers 4 \
    --evaluate_from_checkpoint ${evaluate_from_checkpoint} \
    --calvin_conf_path ${calvin_conf_path} \
    --amp ${amp} \
    --eval_exit_mode ${eval_exit_mode} \
    --multi_execution ${multi_execution} \
    --value_net_ckpt ${value_net_ckpt} \
    --exit_ratio ${exit_ratio} \
    --workers 1 > ${log_file} 2>&1
fi