# D
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --from_scratch \
    --run_name RobotFlamingo_task_D_D-exit-strategy \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --window_size 12 \
    --workers 1 \
    --batch_size_calvin 4 \
    --num_joint_epochs 4 --num_exit_epochs 4  \
    --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
    --joint_lr_scheduler constant --exit_lr_scheduler constant  \
    --joint_learning_rate 1e-4 --exit_lr_scale 0.25 \
    --exit_learning_rate 2.5e-5  \
    --early_exit_layer 11 \
    --precision amp \
    --multi_exit


# ABC
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --from_scratch \
    --run_name RobotFlamingo_task_ABC_D-exit-strategy \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_ABC_D \
    --dataset_resampled \
    --window_size 12 \
    --workers 1 \
    --batch_size_calvin 4 \
    --num_joint_epochs 4 --num_exit_epochs 1  \
    --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
    --joint_lr_scheduler constant --exit_lr_scheduler constant  \
    --joint_learning_rate 1e-4 --exit_lr_scale 0.25 \
    --exit_learning_rate 2.5e-5  \
    --early_exit_layer 11 \
    --precision amp \
    --multi_exit

# ABCD
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --from_scratch \
    --run_name RobotFlamingo_task_ABCD_D-exit-strategy \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_ABCD_D \
    --dataset_resampled \
    --window_size 12 \
    --workers 1 \
    --batch_size_calvin 4 \
    --num_joint_epochs 3 --num_exit_epochs 0  \
    --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
    --joint_lr_scheduler constant --exit_lr_scheduler constant  \
    --joint_learning_rate 1e-4 --exit_lr_scale 0.25 \
    --exit_learning_rate 2.5e-5  \
    --early_exit_layer 11 \
    --precision amp \
    --multi_exit

torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --run_name RobotFlamingo_task_ABCD_D-exit-strategy \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_ABCD_D \
    --dataset_resampled \
    --window_size 12 \
    --workers 1 \
    --batch_size_calvin 4 \
    --num_joint_epochs 3 --num_exit_epochs 1  \
    --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
    --joint_lr_scheduler constant --exit_lr_scheduler constant  \
    --joint_learning_rate 1e-4 --exit_lr_scale 0.25 \
    --exit_learning_rate 2.5e-5  \
    --early_exit_layer 11 \
    --precision amp \
    --multi_exit \
    --resume_from_checkpoint RobotFlamingo_task_ABCD_D-exit-strategy/stg=post_3+0_layer_11_multie_intv=2_mlpdrp=0.4_layerwise_lstmdrp=0.3_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth  \
    --exit_dropout 0.5 \
    --lstm_dropout 0.4


# D 9B
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6047 robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_9b \
    --traj_cons \
    --use_gripper \
    --from_scratch \
    --run_name RobotFlamingo_9B_task_D_D-exit-strategy \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --window_size 12 \
    --workers 1 \
    --batch_size_calvin 6 \
    --num_joint_epochs 4 --num_exit_epochs 4  \
    --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
    --joint_lr_scheduler constant --exit_lr_scheduler constant  \
    --joint_learning_rate 1e-4 --exit_lr_scale 0.25 \
    --exit_learning_rate 2.5e-5  \
    --early_exit_layer 15 \
    --precision amp \
    --multi_exit

# eval
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_ABCD_D-exit-strategy \
            --node_num 8 --eval_exit_mode dynamic --value_type action --threshold_type L2 --amp 1 --exit_dist exp  --exit_ratio 1.001 --num_seq 224 --max_layer 12