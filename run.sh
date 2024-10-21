#  2.27s every iter (2 A100)
# WANDB_MODE=offline torchrun --nnodes=1 --nproc_per_node=1 --master_port=6042 robot_flamingo/train/train_calvin.py \
#     --report_to_wandb \
#     --llm_name mpt_dolly_3b \
#     --traj_cons \
#     --use_gripper \
#     --fusion_mode post \
#     --rgb_pad 10 \
#     --gripper_pad 4 \
#     --precision fp32 \
#     --num_epochs 200 --save_freq 50 \
#     --gradient_accumulation_steps 1 \
#     --batch_size_calvin 6 \
#     --run_name RobotFlamingoDBG \
#     --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/calvin_debug_dataset \
#     --cross_attn_every_n_layers 4 \
#     --dataset_resampled \
#     --loss_multiplier_calvin 1.0 \
#     --workers 1 \
#     --lr_scheduler constant \
#     --warmup_steps 5000 \
#     --learning_rate 1e-4 \
#     --save_every_iter 10000 \
#     --from_scratch \
#     --window_size 12 \
#     --early_exit_layer -1

# #  1.95 every iter (2 A100)
#     WANDB_MODE=offline torchrun --nnodes=1 --nproc_per_node=1 --master_port=6045 robot_flamingo/train/train_calvin.py \
#         --report_to_wandb \
#         --llm_name mpt_dolly_3b \
#         --traj_cons \
#         --use_gripper \
#         --fusion_mode post \
#         --rgb_pad 10 \
#         --gripper_pad 4 \
#         --precision fp32 \
#         --num_epochs 200 --save_freq 25 \
#         --gradient_accumulation_steps 1 \
#         --batch_size_calvin 6 \
#         --run_name RobotFlamingoDBG \
#         --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/calvin_debug_dataset \
#         --cross_attn_every_n_layers 4 \
#         --dataset_resampled \
#         --loss_multiplier_calvin 1.0 \
#         --workers 1 \
#         --lr_scheduler constant \
#         --warmup_steps 5000 \
#         --learning_rate 1e-4 \
#         --save_every_iter 10000 \
#         --from_scratch \
#         --window_size 12 \
#         --early_exit_layer 12

# task_D_D
# @ A100 5epoch on D: bs6 42G 75h; bs8 48G 74h; bs16 70G 74h
torchrun --nnodes=1 --nproc_per_node=4 --master_port=6042 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --precision fp32 \
    --num_epochs 5 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --save_every_iter 10000 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer -1

torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --precision fp32 \
    --num_epochs 5 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --save_every_iter 10000 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer  11


torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --precision fp32 \
    --num_epochs 5 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer  5

# terminated by accident
# torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin.py \
#     --report_to_wandb \
#     --llm_name mpt_dolly_3b \
#     --traj_cons \
#     --use_gripper \
#     --fusion_mode post \
#     --rgb_pad 10 \
#     --gripper_pad 4 \
#     --precision fp32 \
#     --num_epochs 5 --save_freq 1 \
#     --gradient_accumulation_steps 1 \
#     --batch_size_calvin 6 \
#     --run_name RobotFlamingo_task_D_D \
#     --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
#     --dataset_resampled \
#     --loss_multiplier_calvin 1.0 \
#     --workers 1 \
#     --lr_scheduler constant \
#     --warmup_steps 5000 \
#     --learning_rate 1e-4 \
#     --from_scratch \
#     --window_size 12 \
#     --early_exit_layer  2

# reproduce ABC->D and ABCD->D

torchrun --nnodes=1 --nproc_per_node=8 --master_port=6043 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --precision fp32 \
    --num_epochs 8 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_ABC_D \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_ABC_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --save_every_iter 10000 \
    --window_size 12 \
    --early_exit_layer -1 \
    --precision amp

# original: 1 epoch --> 15h; amp + bs=12: 3epoch --> 20h

# lr already is adaptive to bs. No need to change lr when changing bs.
   torchrun --nnodes=1 --nproc_per_node=8 --master_port=6043 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_epochs 3 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_ABCD_D \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_ABCD_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer -1 \
    --precision amp 

python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_D_D --node_num 8
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_D_D-extra-exit --node_num 8 --eval_exit_mode all
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_D_D-exit-strategy --node_num 8 --eval_exit_mode all
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_ABCD_D-exit-strategy --node_num 8 --eval_exit_mode all --num_seq 1000
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_D_D-exit-strategy_L24 --node_num 8 --eval_exit_mode all
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_D_D-exit-strategy --node_num 8 --eval_exit_mode all --layerwise_exit_eval
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_D_D-extra-exit --node_num 8 --eval_exit_mode last

python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_D_D-extra-exit --value_net_ckpt RobotFlamingo_task_D_D-value_net \
            --node_num 8 --eval_exit_mode dynamic --amp --exit_ratio 0.7 1.0

# random
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_D_D-exit-strategy \
            --node_num 8 --eval_exit_mode dynamic --value_type random --amp  --exit_ratio 0.5 1.0 --num_seq 224 --max_layer 12
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_ABCD_D-exit-strategy \
            --node_num 8 --eval_exit_mode dynamic --value_type random --amp  --exit_ratio 0.5 1.0 --num_seq 224 --max_layer 12
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_ABC_D-exit-strategy \
            --node_num 8 --eval_exit_mode dynamic --value_type random --amp  --exit_ratio 0.5 1.0 --num_seq 224 --max_layer 12
# time
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_D_D-exit-strategy \
            --node_num 8 --eval_exit_mode dynamic --value_type time --amp  --exit_ratio 0.6 0.7 1.2 1.4 1.6 2.0 --num_seq 224 --max_layer 12
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_ABCD_D-exit-strategy \
            --node_num 8 --eval_exit_mode dynamic --value_type time --amp  --exit_ratio 0.6 0.7 1.2 1.4 1.6 2.0 --num_seq 224 --max_layer 12
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_ABC_D-exit-strategy \
            --node_num 8 --eval_exit_mode dynamic --value_type time --amp  --exit_ratio 0.6 0.7 1.2 1.4 1.6 2.0 --num_seq 224 --max_layer 12
# feature
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_D_D-exit-strategy \
            --node_num 8 --eval_exit_mode dynamic --value_type sim --amp  --exit_ratio 0.3 0.4 0.5 0.6 0.7 0.8 1.0 1.2 --num_seq 224 --max_layer 12
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_ABCD_D-exit-strategy \
            --node_num 8 --eval_exit_mode dynamic --value_type sim --amp  --exit_ratio 0.3 0.4 0.5 0.6 0.7 0.8 1.0 1.2 --num_seq 224 --max_layer 12
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_ABC_D-exit-strategy \
            --node_num 8 --eval_exit_mode dynamic --value_type sim --amp  --exit_ratio 0.3 0.4 0.5 0.6 0.7 0.8 1.0 1.2 --num_seq 224 --max_layer 12


python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_ABCD_D-exit-strategy \
            --node_num 8 --eval_exit_mode dynamic --value_type action --threshold_type L2 --amp 1 --exit_dist exp  --exit_ratio 3.0 2.0 1.2 1.0 0.7 0.5 0.4 0.3 0.2 0.1 --num_seq 224 --max_layer 12
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_D_D-exit-strategy \
            --node_num 8 --eval_exit_mode dynamic --value_type action --threshold_type L2 --amp 1 --exit_dist exp  --exit_ratio 1.0 0.5 --num_seq 224 --max_layer 12

# 9B
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_9B_task_D_D-exit-strategy --node_num 4 --eval_exit_mode all --num_seq 56 --max_layer 16
# 16 12 8 6 4
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_9B_task_D_D-exit-strategy \
            --node_num $ARNOLD_WORKER_GPU --eval_exit_mode dynamic --value_type action --threshold_type L2 --amp --exit_dist exp  --exit_ratio 0.1 --num_seq 224 --max_layer 8


# baseline++
python3 eval_ckpts.py --ckpt_dir RobotFlamingo++_task_D_D --node_num 8 --eval_exit_mode last --num_seq 224
python3 eval_ckpts.py --ckpt_dir RobotFlamingo++_task_ABC_D --node_num 8 --eval_exit_mode last --num_seq 224
python3 eval_ckpts.py --ckpt_dir RobotFlamingo++_task_ABCD_D --node_num 8 --eval_exit_mode last --num_seq 224
python3 eval_ckpts.py --ckpt_dir RobotFlamingo++_9B_task_D_D --node_num $ARNOLD_WORKER_GPU --eval_exit_mode last --num_seq 224

# bayesian optimization
python3 bayesian_optimization.py --num_seq 224 --acq_func EI --seed 1 --n_calls 20 --init_exit_ratio 0.5 --port 12345 \
    --evaluate_from_checkpoint RobotFlamingo_task_D_D-exit-strategy/stg=post_4+5_layer_11_multie_intv=2_extrae_nodth_reg_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth
python3 bayesian_optimization.py --num_seq 224 --acq_func EI --seed 1 --n_calls 20 --init_exit_ratio 1.0 --port 12345 \
    --evaluate_from_checkpoint RobotFlamingo_task_ABCD_D-exit-strategy/stg=post_3+1_layer_11_multie_intv=2_extrae_nodth_reg_mlpdrp=0.5_layerwise_lstmdrp=0.4_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth
python3 bayesian_optimization.py --num_seq 224 --acq_func EI --seed 1 --n_calls 20 --init_exit_ratio 1.2 --port 12345 \
    --evaluate_from_checkpoint RobotFlamingo_task_ABC_D-exit-strategy/stg=post_4+4_layer_11_multie_intv=2_extrae_nodth_reg_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth

############### NeurIPS rebuttal BEGIN ##############################


# baseline++ with extra loss. eval the last exit instead of the extra exit
python3 eval_ckpts.py --ckpt_dir RobotFlamingo++_task_D_D --node_num 8 --eval_exit_mode last --num_seq 56 --layerwise_exit_eval --amp

# let DeeR have the same performance with RoboFlamingo++ and eval the real LLM inference time
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_ABCD_D-exit-strategy \
            --node_num 1 --eval_exit_mode dynamic --value_type action --threshold_type L2 --exit_dist exp  --exit_ratio 0.5 --num_seq 224 --max_layer 12 --precision fp16

# ABCD->D enrich
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_ABCD_D-exit-strategy --node_num 8 --eval_exit_mode last --amp 1 --num_seq 56 --enrich_annotation 1
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_ABCD_D-exit-strategy \
            --node_num 8 --eval_exit_mode dynamic --value_type action --threshold_type L2 --amp 1 --exit_dist exp  --exit_ratio 1.0 --num_seq 56 --max_layer 12 --enrich_annotation 1


# D->D enrich: DeeR/Flamingo++ L24/12/6 (w. extra exit / layer exit)
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_D_D-exit-strategy_L24 --node_num 8 --eval_exit_mode all --num_seq 56 --enrich_annotation 1
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_D_D-exit-strategy_L24 --node_num 8 --eval_exit_mode all --num_seq 56 --enrich_annotation 1 --layerwise_exit_eval 
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_D_D-exit-strategy --node_num 8 --eval_exit_mode last --num_seq 56 --enrich_annotation 1
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_D_D-exit-strategy \
            --node_num 8 --eval_exit_mode dynamic --value_type action --threshold_type L2 --amp 1 --exit_dist exp  --exit_ratio 2.0 3.0 10.0 0.8 1.0  --num_seq 56 --max_layer 12 --enrich_annotation 1
python3 eval_ckpts.py --ckpt_dir RobotFlamingo++_task_D_D --node_num 8 --eval_exit_mode last --num_seq 56 --enrich_annotation 1 --amp

# quant
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_ABCD_D-exit-strategy \
            --node_num 8 --eval_exit_mode dynamic --value_type action --threshold_type L2 --precision fp16 --exit_dist exp  --exit_ratio 1.0 --num_seq 56 --max_layer 12 
python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_ABCD_D-exit-strategy \
            --node_num 8 --eval_exit_mode dynamic --value_type action --threshold_type L2 --precision int4 --exit_dist exp  --exit_ratio 1.0 --num_seq 56 --max_layer 12 

# solving threshold on unseen/low-data situations
bash threshold.bash

############### NeurIPS rebuttal END ##############################

    torchrun --nnodes=1 --nproc_per_node=$ARNOLD_WORKER_GPU  --master_port=12345 robot_flamingo/eval/eval_calvin.py \
    --precision fp32 \
    --use_gripper \
    --window_size 12 \
    --fusion_mode post \
    --run_name RobotFlamingoDBG \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --cross_attn_every_n_layers 4 \
    --evaluate_from_checkpoint RobotFlamingo_task_D_D-exit-strategy/stg=post_4+5_layer_11_multie_intv=2_extrae_nodth_reg_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth \
    --calvin_conf_path /mnt/bn/yueyang/archive/calvin/calvin_models/conf \
    --amp true \
    --eval_exit_mode dynamic \
    --thresholds 0.030050416706011687 0.0040399048460921325 0.00619066194510008 0.005840546042909084 0.006469010994272101 100000.0 \
    --value_type action \
    --threshold_type L2 --exit_dist exp \
    --num_seq 224 \
    --validation_set \
    --workers 1 > log_BO_0.8_RobotFlamingo_task_D_D-exit-strategy/seq224_test_threshold.log 2>&1

# gaussian
# torchrun --nnodes=1 --nproc_per_node=8 --master_port=6045 robot_flamingo/train/train_calvin.py \
#     --report_to_wandb \
#     --llm_name mpt_dolly_3b \
#     --traj_cons \
#     --use_gripper \
#     --fusion_mode post \
#     --rgb_pad 10 \
#     --gripper_pad 4 \
#     --precision fp32 \
#     --num_epochs 5 --save_freq 1 \
#     --gradient_accumulation_steps 1 \
#     --batch_size_calvin 6 \
#     --run_name RobotFlamingo_task_D_D \
#     --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
#     --dataset_resampled \
#     --loss_multiplier_calvin 1.0 \
#     --workers 1 \
#     --lr_scheduler constant \
#     --warmup_steps 5000 \
#     --learning_rate 1e-4 \
#     --from_scratch \
#     --window_size 12 \
#     --early_exit_layer  5 \
#     --head_type gaussian \
#     --state_dependent_std \
#     --tanh_squash_dist

torchrun --nnodes=1 --nproc_per_node=8 --master_port=6045 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --precision fp32 \
    --num_epochs 5 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer  5 \
    --head_type gaussian \
    --state_dependent_std

torchrun --nnodes=1 --nproc_per_node=8 --master_port=6045 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --precision fp32 \
    --num_epochs 10 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --window_size 12 \
    --early_exit_layer  5 \
    --head_type gaussian \
    --state_dependent_std \
    --bin_coef 1.0

torchrun --nnodes=1 --nproc_per_node=8 --master_port=6045 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --precision fp32 \
    --num_epochs 5 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer  5 \
    --head_type gaussian \
    --state_dependent_std \
    --bin_coef 1.0 \
    --tanh_squash_dist


# proxy

# torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin.py \
#     --report_to_wandb \
#     --llm_name mpt_dolly_3b \
#     --traj_cons \
#     --use_gripper \
#     --fusion_mode post \
#     --rgb_pad 10 \
#     --gripper_pad 4 \
#     --precision fp32 \
#     --num_epochs 5 --save_freq 1 \
#     --gradient_accumulation_steps 1 \
#     --batch_size_calvin 6 \
#     --run_name RobotFlamingo_task_D_D \
#     --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
#     --dataset_resampled \
#     --loss_multiplier_calvin 1.0 \
#     --workers 1 \
#     --lr_scheduler constant \
#     --warmup_steps 5000 \
#     --learning_rate 1e-4 \
#     --from_scratch \
#     --window_size 12 \
#     --early_exit_layer  5 \
#     --data_percent 0.2

torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_epochs 5 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D-subset-v2 \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer  5 \
    --data_percent 0.5 \
    --precision amp

# multi-exit
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_epochs 5 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D-multi-exit \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer  5 \
    --data_percent 0.5 \
    --precision amp \
    --multi_exit

# sweep exit_lr_scale, exit_decay, exit_dropout
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_epochs 5 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D-multi-exit \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer  5 \
    --data_percent 0.5 \
    --precision amp \
    --multi_exit \
    --exit_lr_scale 0.5

torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_epochs 5 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D-multi-exit \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer  5 \
    --data_percent 0.5 \
    --precision amp \
    --multi_exit \
    --exit_lr_scale 0.25

torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_epochs 5 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D-multi-exit \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer  5 \
    --data_percent 0.5 \
    --precision amp \
    --multi_exit \
    --exit_lr_scale 0.1

torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_epochs 5 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D-multi-exit \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer  5 \
    --data_percent 0.5 \
    --precision amp \
    --multi_exit \
    --exit_lr_scale 0.25 \
    --exit_decay

torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_epochs 5 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D-multi-exit \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer  5 \
    --data_percent 0.5 \
    --precision amp \
    --multi_exit \
    --exit_lr_scale 0.25 \
    --exit_dropout 0.1
    # 0.2, 0.5

# multiple action
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_epochs 5 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D-subset-v2 \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer  5 \
    --data_percent 0.5 \
    --precision amp \
    --multi_step_action 3

# train value_net
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_value.py \
    --report_to_wandb \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_epochs 5 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D-value_net \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-3 \
    --from_scratch \
    --window_size 12 \
    --data_percent 0.5 \
    --precision amp \
    --value_weight_decay 0 --value_dropout 0 \
    --roboflamingo_checkpoint RobotFlamingo_task_D_D-extra-exit/amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_extra-exit_lr_scale=0.25_dropout=0.1_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth \
    --discrete

# lr : 3e-6 1e-5 1e-4 1e-3

# train RoboFlamingo with extra exit
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_epochs 5 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D-extra-exit \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer  5 \
    --data_percent 0.5 \
    --precision amp \
    --multi_exit \
    --use_extra_exit \
    --exit_lr_scale 0.25 \
    --exit_dropout 0.1

torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_epochs 5 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo_task_D_D-extra-exit \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler cosine \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer  5 \
    --data_percent 0.5 \
    --precision amp \
    --multi_exit \
    --use_extra_exit \
    --exit_lr_scale 0.25 \
    --exit_dropout 0.1

# 12 layers, exit_interval=2, bs=4, 31G; 8V100, data=1.0, 10h
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_epochs 5 --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 4 \
    --run_name RobotFlamingo_task_D_D-extra-exit \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer 11 \
    --data_percent 1.0 \
    --precision amp \
    --exit_lr_scale 0.25 \
    --exit_dropout 0.1 \
    --multi_exit \
    --exit_interval 2 \
    --use_extra_exit 

# exit train strategy
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_pre_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_exit_epochs 3 --num_joint_epochs 4 \
    --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 4 \
    --run_name RobotFlamingo_task_D_D-exit-strategy \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
    --exit_learning_rate 1e-4 --joint_learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer 11 \
    --data_percent 1.0 \
    --precision amp \
    --exit_lr_scale 0.25 \
    --exit_dropout 0.1 \
    --multi_exit \
    --exit_interval 2 \
    --use_extra_exit 

torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_exit_epochs 6 --num_joint_epochs 4 \
    --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 4 \
    --run_name RobotFlamingo_task_D_D-exit-strategy \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
    --exit_learning_rate 2.5e-5 --joint_learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer 11 \
    --data_percent 1.0 \
    --precision amp \
    --exit_lr_scale 0.25 \
    --exit_dropout 0.1 \
    --multi_exit \
    --exit_interval 2 \
    --use_extra_exit 

# torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
#     --report_to_wandb \
#     --llm_name mpt_dolly_3b \
#     --traj_cons \
#     --use_gripper \
#     --fusion_mode post \
#     --rgb_pad 10 \
#     --gripper_pad 4 \
#     --num_exit_epochs 6 --num_joint_epochs 5 \
#     --resume_from_checkpoint RobotFlamingo_task_D_D-exit-strategy/strategy=post_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth \
#     --save_freq 1 \
#     --gradient_accumulation_steps 1 \
#     --batch_size_calvin 4 \
#     --run_name RobotFlamingo_task_D_D-exit-strategy \
#     --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
#     --dataset_resampled \
#     --loss_multiplier_calvin 1.0 \
#     --workers 1 \
#     --lr_scheduler constant \
#     --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
#     --exit_learning_rate 2.5e-5 --joint_learning_rate 1e-4 \
#     --window_size 12 \
#     --early_exit_layer 11 \
#     --data_percent 1.0 \
#     --precision amp \
#     --exit_lr_scale 0.25 \
#     --exit_dropout 0.1 \
#     --multi_exit \
#     --exit_interval 2 \
#     --use_extra_exit \
#     --wandb_note reumse_post_4+0->5+6


# torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
#     --report_to_wandb \
#     --llm_name mpt_dolly_3b \
#     --traj_cons \
#     --use_gripper \
#     --fusion_mode post \
#     --rgb_pad 10 \
#     --gripper_pad 4 \
#     --num_exit_epochs 6 --num_joint_epochs 3 \
#     --resume_from_checkpoint RobotFlamingo_task_D_D-exit-strategy/strategy=post_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth \
#     --save_freq 1 \
#     --gradient_accumulation_steps 1 \
#     --batch_size_calvin 4 \
#     --run_name RobotFlamingo_task_D_D-exit-strategy \
#     --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
#     --dataset_resampled \
#     --loss_multiplier_calvin 1.0 \
#     --workers 1 \
#     --lr_scheduler constant \
#     --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
#     --exit_learning_rate 2.5e-5 --joint_learning_rate 1e-4 \
#     --window_size 12 \
#     --early_exit_layer 11 \
#     --data_percent 1.0 \
#     --precision amp \
#     --exit_lr_scale 0.25 \
#     --exit_dropout 0.1 \
#     --multi_exit \
#     --exit_interval 2 \
#     --use_extra_exit \
    # --wandb_note reumse_post_3+0->3+6


torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_exit_epochs 5 --num_joint_epochs 4 \
    --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 4 \
    --run_name RobotFlamingo_task_D_D-exit-strategy \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
    --exit_learning_rate 2.5e-5 --joint_learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer 11 \
    --data_percent 1.0 \
    --precision amp \
    --exit_lr_scale 0.25 \
    --exit_dropout 0.1 \
    --multi_exit \
    --exit_interval 2 \
    --use_extra_exit  \
    --mlp_layernorm

torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_exit_epochs 5 --num_joint_epochs 4 \
    --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 4 \
    --run_name RobotFlamingo_task_D_D-exit-strategy \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
    --exit_learning_rate 2.5e-5 --joint_learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 \
    --early_exit_layer 11 \
    --data_percent 1.0 \
    --precision amp \
    --exit_lr_scale 0.25 \
    --exit_dropout 0.1 \
    --multi_exit \
    --exit_interval 2 \
    --use_extra_exit  \
    --mlp_layernorm \
    --lstm_layernorm

# dropout
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_exit_epochs 5 --num_joint_epochs 4 \
    --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 4 \
    --run_name RobotFlamingo_task_D_D-exit-strategy \
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
    --lstm_num_layers 4

# improve extra exit
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_exit_epochs 5 --num_joint_epochs 4 \
    --save_freq 1 \
    --from_scratch \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 4 \
    --run_name RobotFlamingo_task_D_D-exit-strategy \
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

# ablation: no auxiliary action head
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_exit_epochs 5 --num_joint_epochs 4 \
    --save_freq 1 \
    --from_scratch \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 4 \
    --run_name RobotFlamingo_task_D_D-exit-strategy \
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
    --regularize_extra_exit \
    --no_auxiliary_action_head_loss

# freeze embed for enriched setting D->D
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_exit_epochs 5 --num_joint_epochs 4 \
    --save_freq 1 \
    --from_scratch \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 4 \
    --run_name RobotFlamingo_task_D_D-exit-strategy \
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
    --regularize_extra_exit \
    --no_auxiliary_action_head_loss \
    --freeze_embed

# robofalmingo++ 3B D

torchrun --nnodes=$ARNOLD_WORKER_NUM  --node_rank=$ARNOLD_ID --master_addr=$METIS_WORKER_0_HOST --nproc_per_node=$ARNOLD_WORKER_GPU --master_port=$METIS_WORKER_0_PORT robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_joint_epochs 4 --num_exit_epochs 4  \
    --save_freq 1 \
    --from_scratch \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo++_task_D_D \
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
    --early_exit_layer 11 \
    --data_percent 1.0 \
    --precision amp \
    --mlp_layernorm \
    --lstm_layernorm \
    --exit_dropout 0.4 \
    --lstm_dropout 0.3 \
    --dropout_mode layerwise \
    --mlp_num_hidden_layers 2 \
    --lstm_num_layers 4

# add auxiliary loss at intermediate layers for roboflamingo++
# Technically achieve it by enabling the extra exit (share / no share the extra exit)
# the shared version also serves as measuring the training cost of AuxHead (AuxLoss)
torchrun --nnodes=$ARNOLD_WORKER_NUM  --node_rank=$ARNOLD_ID --master_addr=$METIS_WORKER_0_HOST --nproc_per_node=$ARNOLD_WORKER_GPU --master_port=$METIS_WORKER_0_PORT robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_joint_epochs 4 --num_exit_epochs 4  \
    --save_freq 1 \
    --from_scratch \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo++_task_D_D \
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
    --data_percent 1.0 \
    --precision amp \
    --mlp_layernorm \
    --lstm_layernorm \
    --exit_dropout 0.4 \
    --lstm_dropout 0.3 \
    --dropout_mode layerwise \
    --mlp_num_hidden_layers 2 \
    --lstm_num_layers 4 \
    --early_exit_layer 11 \
    --use_extra_exit \
    --regularize_extra_exit \
    --share_exit



torchrun --nnodes=$ARNOLD_WORKER_NUM  --node_rank=$ARNOLD_ID --master_addr=$METIS_WORKER_0_HOST --nproc_per_node=$ARNOLD_WORKER_GPU --master_port=$METIS_WORKER_0_PORT robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_joint_epochs 4 --num_exit_epochs 4  \
    --save_freq 1 \
    --from_scratch \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo++_task_D_D \
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
    --data_percent 1.0 \
    --precision amp \
    --mlp_layernorm \
    --lstm_layernorm \
    --exit_dropout 0.4 \
    --lstm_dropout 0.3 \
    --dropout_mode layerwise \
    --mlp_num_hidden_layers 2 \
    --lstm_num_layers 4 \
    --early_exit_layer 11 \
    --use_extra_exit \
    --regularize_extra_exit 


# robofalmingo++ 3B ABC
torchrun --nnodes=$ARNOLD_WORKER_NUM  --node_rank=$ARNOLD_ID --master_addr=$METIS_WORKER_0_HOST --nproc_per_node=$ARNOLD_WORKER_GPU --master_port=$METIS_WORKER_0_PORT robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_joint_epochs 4  --num_exit_epochs 1 \
    --save_freq 1 \
    --from_scratch \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo++_task_ABC_D \
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
    --early_exit_layer 23 \
    --data_percent 1.0 \
    --precision amp \
    --mlp_layernorm \
    --lstm_layernorm \
    --exit_dropout 0.4 \
    --lstm_dropout 0.3 \
    --dropout_mode layerwise \
    --mlp_num_hidden_layers 2 \
    --lstm_num_layers 4

torchrun --nnodes=$ARNOLD_WORKER_NUM  --node_rank=$ARNOLD_ID --master_addr=$METIS_WORKER_0_HOST --nproc_per_node=$ARNOLD_WORKER_GPU --master_port=$METIS_WORKER_0_PORT robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
   --num_joint_epochs 3  --num_exit_epochs 0  \
    --save_freq 1 \
    --from_scratch \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo++_task_ABCD_D \
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
    --early_exit_layer 23 \
    --data_percent 1.0 \
    --precision amp \
    --mlp_layernorm \
    --lstm_layernorm \
    --exit_dropout 0.4 \
    --lstm_dropout 0.3 \
    --dropout_mode layerwise \
    --mlp_num_hidden_layers 2 \
    --lstm_num_layers 4

# # resume post 1 epoch
torchrun --nnodes=$ARNOLD_WORKER_NUM  --node_rank=$ARNOLD_ID --master_addr=$METIS_WORKER_0_HOST --nproc_per_node=$ARNOLD_WORKER_GPU --master_port=$METIS_WORKER_0_PORT robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
   --num_joint_epochs 3  --num_exit_epochs 1  \
    --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingo++_task_ABCD_D \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_ABCD_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --exit_warmup_steps 2500 --joint_warmup_steps 2500 \
    --joint_lr_scheduler constant --exit_lr_scheduler constant  \
    --joint_learning_rate 1e-4 --exit_lr_scale 0.25 \
    --exit_learning_rate 2.5e-5  \
    --resume_from_checkpoint RobotFlamingo++_task_ABCD_D/stg=post_3+0_layer_5_mlpdrp=0.4_layerwise_lstmdrp=0.3_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth \
    --window_size 12 \
    --early_exit_layer 5 \
    --data_percent 1.0 \
    --precision amp \
    --mlp_layernorm \
    --lstm_layernorm \
    --exit_dropout 0.5 \
    --lstm_dropout 0.4 \
    --dropout_mode layerwise \
    --mlp_num_hidden_layers 2 \
    --lstm_num_layers 4

# RoboFlamingo++ 9B
torchrun --nnodes=$ARNOLD_WORKER_NUM  --node_rank=$ARNOLD_ID --master_addr=$METIS_WORKER_0_HOST --nproc_per_node=$ARNOLD_WORKER_GPU --master_port=$METIS_WORKER_0_PORT robot_flamingo/train/train_calvin_post_strategy.py \
    --report_to_wandb \
    --llm_name mpt_9b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --num_joint_epochs 4  --num_exit_epochs 0  \
    --save_freq 1 \
    --from_scratch \
    --gradient_accumulation_steps 1 \
    --run_name RobotFlamingo++_9B_task_D_D \
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
    --batch_size_calvin 4 \
    --data_percent 1.0 \
    --precision amp \
    --mlp_layernorm \
    --lstm_layernorm \
    --exit_dropout 0.4 \
    --lstm_dropout 0.3 \
    --dropout_mode layerwise \
    --mlp_num_hidden_layers 2 \
    --lstm_num_layers 4


# torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_post_strategy.py \
#     --report_to_wandb \
#     --llm_name mpt_dolly_3b \
#     --traj_cons \
#     --use_gripper \
#     --fusion_mode post \
#     --rgb_pad 10 \
#     --gripper_pad 4 \
#     --num_exit_epochs 5 --num_joint_epochs 4 \
#     --save_freq 1 \
#     --gradient_accumulation_steps 1 \
#     --batch_size_calvin 4 \
#     --run_name RobotFlamingo_task_D_D-exit-strategy \
#     --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
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
#     --wandb_note layerwise_projection \
#     # --wandb_note fix_index \


torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_joint_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --llm_update_freq 2 \
    --num_joint_epochs 10 \
    --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 4 \
    --run_name RobotFlamingo_task_D_D-exit-strategy \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --joint_warmup_steps 2500 \
    --learning_rate 1e-4 \
    --window_size 12 \
    --early_exit_layer 11 \
    --data_percent 1.0 \
    --precision amp \
    --exit_lr_scale 0.25 \
    --exit_dropout 0.1 \
    --multi_exit \
    --exit_interval 2 \
    --use_extra_exit 

torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin_joint_strategy.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --llm_update_freq 3 \
    --num_joint_epochs 12 \
    --resume_from_checkpoint RobotFlamingo_task_D_D-exit-strategy/strategy=joint_frq=3_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth \
    --save_freq 1 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 4 \
    --run_name RobotFlamingo_task_D_D-exit-strategy \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --joint_warmup_steps 2500 \
    --learning_rate 1e-4 \
    --window_size 12 \
    --early_exit_layer 11 \
    --data_percent 1.0 \
    --precision amp \
    --exit_lr_scale 0.25 \
    --exit_dropout 0.1 \
    --multi_exit \
    --exit_interval 2 \
    --use_extra_exit 

# distill loss (1.0, 0.1, 0.01, 0.001, 0)
# torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_calvin.py \
#     --report_to_wandb \
#     --llm_name mpt_dolly_3b \
#     --traj_cons \
#     --use_gripper \
#     --fusion_mode post \
#     --rgb_pad 10 \
#     --gripper_pad 4 \
#     --num_epochs 5 --save_freq 1 \
#     --gradient_accumulation_steps 1 \
#     --batch_size_calvin 6 \
#     --run_name RobotFlamingo_task_D_D-extra-exit \
#     --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
#     --dataset_resampled \
#     --loss_multiplier_calvin 1.0 \
#     --workers 1 \
#     --lr_scheduler constant \
#     --warmup_steps 5000 \
#     --learning_rate 1e-4 \
#     --from_scratch \
#     --window_size 12 \
#     --early_exit_layer  5 \
#     --data_percent 0.5 \
#     --precision amp \
#     --multi_exit \
#     --use_extra_exit \
#     --exit_lr_scale 0.25 \
#     --exit_dropout 0.1 \
#     --feat_distill_coef 0.001 \
#     --wandb_note proj2_distill1e-3
#     # --wandb_note max_pool_feat_distill



# debug for training discrete value net

# # 1e-4 1e-5
# torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_value.py \
#     --report_to_wandb \
#     --traj_cons \
#     --use_gripper \
#     --fusion_mode post \
#     --rgb_pad 10 \
#     --gripper_pad 4 \
#     --num_epochs 5 --save_freq 1 \
#     --gradient_accumulation_steps 1 \
#     --batch_size_calvin 16 \
#     --run_name RobotFlamingo_task_D_D-value_net-v2 \
#     --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
#     --dataset_resampled \
#     --loss_multiplier_calvin 1.0 \
#     --workers 1 \
#     --lr_scheduler constant \
#     --warmup_steps 2500 \
#     --learning_rate 1e-4 \
#     --from_scratch \
#     --window_size 12 \
#     --data_percent 0.2 \
#     --precision amp \
#     --value_weight_decay 0 \
#     --roboflamingo_checkpoint RobotFlamingo_task_D_D-exit-strategy/fix_index_strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_nodetach_mlp2L_mlpln_lstmln_mlpdrp=0.4_layerwise_lstmdrp=0.3_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_jointlr_0.000067_exitscale=0.25_7.pth \
#     --discrete --num_bin 20

# # train jointly with Flamingo Backbone
# torchrun --nnodes=1 --nproc_per_node=8 --master_port=6046 robot_flamingo/train/train_value.py \
#     --report_to_wandb \
#     --traj_cons \
#     --use_gripper \
#     --fusion_mode post \
#     --rgb_pad 10 \
#     --gripper_pad 4 \
#     --save_freq 2 \
#     --gradient_accumulation_steps 1 \
#     --batch_size_calvin 6 \
#     --run_name RobotFlamingo_task_D_D-value_net \
#     --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
#     --dataset_resampled \
#     --loss_multiplier_calvin 1.0 \
#     --workers 1 \
#     --lr_scheduler constant \
#     --warmup_steps 5000 \
#     --learning_rate 1e-4 \
#     --from_scratch \
#     --window_size 12 \
#     --data_percent 0.1 \
#     --precision amp \
#     --value_weight_decay 0 --value_dropout 0 \
#     --roboflamingo_checkpoint RobotFlamingo_task_D_D-extra-exit/amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_extra-exit_lr_scale=0.25_dropout=0.1_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth \
#     --discrete --num_bin 2 --num_epochs 5 \
#     --value_net_lr_scale 0.25 --exit_lr_scale 0.25




    # count param and flops
    python3 eval_ckpts.py --ckpt_dir RobotFlamingo_task_D_D --node_num 1

torchrun --nnodes=$ARNOLD_WORKER_NUM  --node_rank=$ARNOLD_ID --master_addr=$METIS_WORKER_0_HOST --nproc_per_node=1 --master_port=12345 robot_flamingo/eval/eval_calvin.py \
    --precision fp32 \
    --use_gripper \
    --window_size 12 \
    --fusion_mode post \
    --run_name RobotFlamingoDBG \
    --batch_size_calvin 6 \
    --data_percent 0.002 \
    --validation_set \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --evaluate_from_checkpoint RobotFlamingo_9B_task_D_D/amp_checkpoint_gripper_post_hist_1__exit_layer_-1_aug_10_4_traj_cons_ws_12_mpt_9b_0.pth0.pth \
    --calvin_conf_path /mnt/bn/yueyang/archive/calvin/calvin_models/conf \
    --amp True \
    --eval_exit_mode last \
    --workers 1
