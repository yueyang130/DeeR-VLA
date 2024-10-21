# D L24


torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_joint_epochs 4 --num_exit_epochs 5  \
    --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D-exit-strategy_L24 \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
    --joint_lr_scheduler constant --exit_lr_scheduler constant  \
    --joint_learning_rate 1e-4 --exit_lr_scale 0.25 \
    --exit_learning_rate 2.5e-5  \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer 23 \
    --data_percent 1.0 \
    --precision amp \
    --multi_exit \
    --exit_interval 2 \
    --use_extra_exit  \
    --mlp_layernorm \
    --lstm_layernorm \
    --exit_dropout 0.4 \
    --lstm_dropout 0.3 \
    --dropout_mode layerwise \
    --mlp_num_hidden_layers 2 \
    --lstm_num_layers 4 \
    --detach_extra_exit 0 \
    --regularize_extra_exit


#################################################################################################################

torchrun --nnodes=2 --node_rank=0 --master_addr=10.128.101.150 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_joint_epochs 3 --num_exit_epochs 0  \
    --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 4 \
    --run_name RobotFlamingo_task_ABCD_D-exit-strategy \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_ABCD_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
    --joint_lr_scheduler constant --exit_lr_scheduler constant  \
    --joint_learning_rate 1e-4 --exit_lr_scale 0.25 \
    --exit_learning_rate 2.5e-5  \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer 11 \
    --data_percent 1.0 \
    --precision amp \
    --multi_exit \
    --exit_interval 2 \
    --use_extra_exit  \
    --mlp_layernorm \
    --lstm_layernorm \
    --exit_dropout 0.4 \
    --lstm_dropout 0.3 \
    --dropout_mode layerwise \
    --mlp_num_hidden_layers 2 \
    --lstm_num_layers 4 \
    --detach_extra_exit 0 \
    --regularize_extra_exit

# increase dropout in the second stage
torchrun --nnodes=2 --node_rank=1 --master_addr=10.130.25.22 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_joint_epochs 3 --num_exit_epochs 1  \
    --save_every_iter 2500 \
    --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 4 \
    --run_name RobotFlamingo_task_ABCD_D-exit-strategy \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_ABCD_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
    --joint_lr_scheduler constant --exit_lr_scheduler constant  \
    --joint_learning_rate 1e-4 --exit_lr_scale 0.25 \
    --exit_learning_rate 2.5e-5  \
    --resume_from_checkpoint RobotFlamingo_task_ABCD_D-exit-strategy/stg=post_3+3_layer_11_multie_intv=2_extrae_nodth_reg_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth \
    --window_size 12 \
    --early_exit_layer 11 \
    --data_percent 1.0 \
    --precision amp \
    --multi_exit \
    --exit_interval 2 \
    --use_extra_exit  \
    --mlp_layernorm \
    --lstm_layernorm \
    --exit_dropout 0.5 \
    --lstm_dropout 0.4 \
    --dropout_mode layerwise \
    --mlp_num_hidden_layers 2 \
    --lstm_num_layers 4 \
    --detach_extra_exit 0 \
    --regularize_extra_exit


######## L12/24 ABCD->D (no_auxiliary_action_head_loss)

# torchrun --nnodes=$ARNOLD_WORKER_NUM  --node_rank=$ARNOLD_ID --master_addr=$METIS_WORKER_0_HOST --nproc_per_node=$ARNOLD_WORKER_GPU --master_port=$METIS_WORKER_0_PORT robot_flamingo/train/train_calvin_post_strategy.py \
#     --report_to_wandb \
#     --llm_name mpt_dolly_3b \
#     --traj_cons \
#     --use_gripper \
#     --fusion_mode post \
#     --rgb_pad 10 \
#     --gripper_pad 4 \
#     --num_joint_epochs 3 --num_exit_epochs 0  \
#     --save_freq 1 \
#     --gradient_accumulation_steps 1 \
#     --batch_size_calvin 4 \
#     --run_name RobotFlamingo_task_ABCD_D-exit-strategy \
#     --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_ABCD_D \
#     --dataset_resampled \
#     --loss_multiplier_calvin 1.0 \
#     --workers 1 \
#     --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
#     --joint_lr_scheduler constant --exit_lr_scheduler constant  \
#     --joint_learning_rate 1e-4 --exit_lr_scale 0.25 \
#     --exit_learning_rate 2.5e-5  \
#     --from_scratch \
#     --window_size 12 \
#     --early_exit_layer 11 \
#     --data_percent 1.0 \
#     --precision amp \
#     --multi_exit \
#     --exit_interval 2 \
#     --use_extra_exit  \
#     --mlp_layernorm \
#     --lstm_layernorm \
#     --exit_dropout 0.4 \
#     --lstm_dropout 0.3 \
#     --dropout_mode layerwise \
#     --mlp_num_hidden_layers 2 \
#     --lstm_num_layers 4 \
#     --detach_extra_exit 0 \
#     --regularize_extra_exit \
#     --no_auxiliary_action_head_loss

# torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
#     --report_to_wandb \
#     --llm_name mpt_dolly_3b \
#     --traj_cons \
#     --use_gripper \
#     --fusion_mode post \
#     --rgb_pad 10 \
#     --gripper_pad 4 \
#     --num_joint_epochs 3 --num_exit_epochs 0  \
#     --save_freq 1 \
#     --gradient_accumulation_steps 1 \
#     --batch_size_calvin 6 \
#     --run_name RobotFlamingo_task_ABCD_D-exit-strategy_L24 \
#     --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_ABCD_D \
#     --dataset_resampled \
#     --loss_multiplier_calvin 1.0 \
#     --workers 1 \
#     --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
#     --joint_lr_scheduler constant --exit_lr_scheduler constant  \
#     --joint_learning_rate 1e-4 --exit_lr_scale 0.25 \
#     --exit_learning_rate 2.5e-5  \
#     --from_scratch \
#     --window_size 12 \
#     --early_exit_layer 23 \
#     --data_percent 1.0 \
#     --precision amp \
#     --multi_exit \
#     --exit_interval 2 \
#     --use_extra_exit  \
#     --mlp_layernorm \
#     --lstm_layernorm \
#     --exit_dropout 0.4 \
#     --lstm_dropout 0.3 \
#     --dropout_mode layerwise \
#     --mlp_num_hidden_layers 2 \
#     --lstm_num_layers 4 \
#     --detach_extra_exit 0 \
#     --regularize_extra_exit \
#     --no_auxiliary_action_head_loss


####################################################################################################################


torchrun --nnodes=2 --node_rank=0 --master_addr=10.130.22.23 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_joint_epochs 4 --num_exit_epochs 4  \
    --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_ABC_D-exit-strategy \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_ABC_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
    --joint_lr_scheduler constant --exit_lr_scheduler constant  \
    --joint_learning_rate 1e-4 --exit_lr_scale 0.25 \
    --exit_learning_rate 2.5e-5  \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer 11 \
    --data_percent 1.0 \
    --precision amp \
    --multi_exit \
    --exit_interval 2 \
    --use_extra_exit  \
    --mlp_layernorm \
    --lstm_layernorm \
    --exit_dropout 0.4 \
    --lstm_dropout 0.3 \
    --dropout_mode layerwise \
    --mlp_num_hidden_layers 2 \
    --lstm_num_layers 4 \
    --detach_extra_exit 0 \
    --regularize_extra_exit

torchrun --nnodes=2 --node_rank=1 --master_addr=10.130.22.23 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_joint_epochs 4 --num_exit_epochs 4  \
    --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 4 \
    --run_name RobotFlamingo_task_ABC_D-exit-strategy \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_ABC_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
    --joint_lr_scheduler constant --exit_lr_scheduler constant  \
    --joint_learning_rate 1e-4 --exit_lr_scale 0.25 \
    --exit_learning_rate 2.5e-5  \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer 11 \
    --data_percent 1.0 \
    --precision amp \
    --multi_exit \
    --exit_interval 2 \
    --use_extra_exit  \
    --mlp_layernorm \
    --lstm_layernorm \
    --exit_dropout 0.4 \
    --lstm_dropout 0.3 \
    --dropout_mode layerwise \
    --mlp_num_hidden_layers 2 \
    --lstm_num_layers 4 \
    --detach_extra_exit 0 \
    --regularize_extra_exit


# train 9B

torchrun --nnodes=$ARNOLD_WORKER_NUM  --node_rank=$ARNOLD_ID --master_addr=$METIS_WORKER_0_HOST --nproc_per_node=$ARNOLD_WORKER_GPU --master_port=$METIS_WORKER_0_PORT robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_9b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_joint_epochs 4 --num_exit_epochs 4  \
    --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_9B_task_D_D-exit-strategy \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
    --joint_lr_scheduler constant --exit_lr_scheduler constant  \
    --joint_learning_rate 1e-4 --exit_lr_scale 0.25 \
    --exit_learning_rate 2.5e-5  \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer 15 \
    --data_percent 1.0 \
    --precision amp \
    --multi_exit \
    --exit_interval 2 \
    --use_extra_exit  \
    --mlp_layernorm \
    --lstm_layernorm \
    --exit_dropout 0.4 \
    --lstm_dropout 0.3 \
    --dropout_mode layerwise \
    --mlp_num_hidden_layers 2 \
    --lstm_num_layers 4 \
    --detach_extra_exit 0 \
    --regularize_extra_exit