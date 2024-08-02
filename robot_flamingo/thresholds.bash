# load exist threshold
torchrun --nnodes=1 --nproc_per_node=8  --master_port=12345 robot_flamingo/eval/eval_calvin.py \
    --precision fp16 \
    --use_gripper \
    --window_size 12 \
    --fusion_mode post \
    --run_name RobotFlamingoDBG \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --validation_set \
    --data_percent 0.1 \
    --load_threshold 1 \
    --cross_attn_every_n_layers 4 \
    --evaluate_from_checkpoint RobotFlamingo_task_ABCD_D-exit-strategy/stg=post_3+1_layer_11_multie_intv=2_extrae_nodth_reg_mlpdrp=0.5_layerwise_lstmdrp=0.4_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth \
    --calvin_conf_path /mnt/bn/yueyang/archive/calvin/calvin_models/conf \
    --eval_exit_mode dynamic \
    --exit_ratio 1.0 \
    --value_type action \
    --threshold_type L2 --exit_dist exp --max_layer 12 \
    --num_seq 56 \
    --workers 1 > log_solving_threshold_ablation/ABC_D_solve_on_valD_1.5 2>&1

# original: solve threshold on D val set
torchrun --nnodes=1 --nproc_per_node=8  --master_port=12345 robot_flamingo/eval/eval_calvin.py \
    --precision fp16 \
    --use_gripper \
    --window_size 12 \
    --fusion_mode post \
    --run_name RobotFlamingoDBG \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
    --validation_set \
    --data_percent 0.1 \
    --load_threshold 0 \
    --cross_attn_every_n_layers 4 \
    --evaluate_from_checkpoint RobotFlamingo_task_ABCD_D-exit-strategy/stg=post_3+1_layer_11_multie_intv=2_extrae_nodth_reg_mlpdrp=0.5_layerwise_lstmdrp=0.4_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth \
    --calvin_conf_path /mnt/bn/yueyang/archive/calvin/calvin_models/conf \
    --eval_exit_mode dynamic \
    --exit_ratio 1.0 \
    --value_type action \
    --threshold_type L2 --exit_dist exp --max_layer 12 \
    --num_seq 56 \
    --workers 1 > log_solving_threshold_ablation/ABC_D_solve_on_valD_1.5 2>&1

# solve threshold on ABC training set
torchrun --nnodes=1 --nproc_per_node=8  --master_port=12345 robot_flamingo/eval/eval_calvin.py \
    --precision fp16 \
    --use_gripper \
    --window_size 12 \
    --fusion_mode post \
    --run_name RobotFlamingoDBG \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_ABC_D \
    --data_percent 0.001 \
    --load_threshold 0 \
    --cross_attn_every_n_layers 4 \
    --evaluate_from_checkpoint RobotFlamingo_task_ABC_D-exit-strategy/stg=post_4+4_layer_11_multie_intv=2_extrae_nodth_reg_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth \
    --calvin_conf_path /mnt/bn/yueyang/archive/calvin/calvin_models/conf \
    --eval_exit_mode dynamic \
    --exit_ratio 1.0 \
    --value_type action \
    --threshold_type L2 --exit_dist exp --max_layer 12 \
    --num_seq 224 \
    --workers 1 > log_solving_threshold_ablation/ABC_D_solve_on_train_0.001_alpha=1.0_seq224 2>&1

torchrun --nnodes=1 --nproc_per_node=8  --master_port=12346 robot_flamingo/eval/eval_calvin.py \
    --precision fp16 \
    --use_gripper \
    --window_size 12 \
    --fusion_mode post \
    --run_name RobotFlamingoDBG \
    --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_ABC_D \
    --data_percent 0.01 \
    --load_threshold 0 \
    --cross_attn_every_n_layers 4 \
    --evaluate_from_checkpoint RobotFlamingo_task_ABC_D-exit-strategy/stg=post_4+4_layer_11_multie_intv=2_extrae_nodth_reg_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth \
    --calvin_conf_path /mnt/bn/yueyang/archive/calvin/calvin_models/conf \
    --eval_exit_mode dynamic \
    --exit_ratio 1.5 \
    --value_type action \
    --threshold_type L2 --exit_dist exp --max_layer 12 \
    --num_seq 56 \
    --workers 1 > log_solving_threshold_ablation/ABC_D_solve_on_trainABC_0.01_alpha=1.5 2>&1

# solve threshold on D training set
