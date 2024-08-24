import os
import glob
import argparse

# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
parser.add_argument('--ckpt_dir', type=str, help='The checkpoint directory')
parser.add_argument('--value_net_ckpt_dir', type=str, default='', help='The checkpoint directory')
parser.add_argument('--exit_ratio', nargs='+', type=float, default=[1.0], help='a list')
parser.add_argument('--single_step', action='store_true', help='If set, evlauate in single step mode')
parser.add_argument('--node_num', type=int)
parser.add_argument("--num_seq", type=int, default=56, help="how many actions are executed in one time when predicting multiple actions; if only one predicted action, repeat it K times")

parser.add_argument(
        "--amp",
        default=0,
        type=int,
    )
parser.add_argument("--multi_execution", type=int, default=1, help="how many actions are executed in one time when predicting multiple actions; if only one predicted action, repeat it K times")
parser.add_argument("--layerwise_exit_eval", action='store_true', default=False) 

parser.add_argument("--eval_exit_mode", type=str, default='last', choices=['last', 'all', 'dynamic']) # only eval the last exit / all exits / dynamic early-exit mechanism
parser.add_argument("--value_type", type=str, default='loss', choices=['loss', 'sim', 'time', 'random', 'action']) # only eval the last exit / all exits / dynamic early-exit mechanism
parser.add_argument("--threshold_type", type=str, default='mean', choices=['mean', 'L2', 'max']) 
parser.add_argument("--exit_dist", type=str, default='exp', choices=['exp', 'gauss', 'gamma']) 
parser.add_argument("--use_action_ensemble", type=int, default=0)
parser.add_argument("--max_layer", type=int, default=-1) # use for constraining memory/max flop. 

parser.add_argument('--enrich_annotation', type=int, default=0, help='If set, eval in enriched annotation setting')
parser.add_argument(
    "--precision",
    choices=["int4", "int8", "bf16", "fp16", "fp32"],
    default="fp32",
    help="Floating point precision.",
)

parser.add_argument("--note", type=str, default='')


# Parse the arguments
args = parser.parse_args()
args.layerwise_exit_eval = 1 if args.layerwise_exit_eval else 0

search_path = os.path.join(args.ckpt_dir,  r'*_[0-9].pth')
ckpt_names = [os.path.basename(path) for path in glob.glob(search_path)]
ckpt_names.sort(reverse=True)
ckpt_names = ckpt_names[:1]
ckpt_names = [
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_extra-exit_lr_scale=0.25_dropout=0.1_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth',
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_distill=0.0_extra-exit_lr_scale=0.25_dropout=0.1_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth',
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_distill=0.0_extra-exit_lr_scale=0.25_dropout=0.1_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth',
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_distill=0.001_extra-exit_lr_scale=0.25_dropout=0.1_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth',
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_distill=0.01_extra-exit_lr_scale=0.25_dropout=0.1_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth',
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_distill=0.001_extra-exit_lr_scale=0.25_dropout=0.1_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth',
    # 'proj2_distill0amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_distill=0.0_extra-exit_lr_scale=0.25_dropout=0.1_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth',
    # 'proj2_distill1e-3amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_distill=0.001_extra-exit_lr_scale=0.25_dropout=0.1_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth',
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_distill=0.001_extra-exit_lr_scale=0.25_dropout=0.1_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth',
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_11_multi-exit_uniform_interval=2_distill=0.0_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth',
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_distill=0.0_extra-exit_lr_scale=0.25_dropout=0.1_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_cosine_4.pth',
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_11_multi-exit_uniform_interval=2_distill=0.0_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_5.pth',
    # 'strategy=post_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth',
    # 'strategy=post_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth',
    # 'strategy=post_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_5.pth',
    # 'strategy=post_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_6.pth',
    # 'reumse_post_4+3_strategy=post_4+6_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'reumse_post_4+3_strategy=post_4+6_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'reumse_post_4+3_strategy=post_4+6_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_9.pth',
    # 'reumse_post_4+0-_strategy=post_5+6_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'reumse_post_4+0-_strategy=post_5+6_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'reumse_post_3+0-_strategy=post_3+6_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_5.pth',
    # 'reumse_post_3+0-_strategy=post_3+6_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_6.pth',
    # 'strategy=pre_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth',
    # 'strategy=pre_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_5.pth',
    # 'strategy=pre_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_6.pth',
    # 'strategy=joint_frq=2_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth',
    # 'strategy=joint_frq=3_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth',
    # 'strategy=joint_frq=3_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth',
    # 'strategy=joint_frq=3_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_5.pth',
    # 'strategy=joint_frq=3_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_6.pth',
    # 'strategy=joint_frq=3_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=joint_frq=3_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=joint_frq=3_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_9.pth',
    # 'strategy=joint_frq=3_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_10.pth',
    # 'strategy=post_3+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth',
    # 'strategy=post_3+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth',
    # 'strategy=post_3+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth',
    # 'strategy=post_3+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_5.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_5.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_6.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_5.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_6.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.1_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_6.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.2_last_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_6.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.1_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.2_last_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.3_last_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.2_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.3_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.1_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.2_last_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.3_last_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.2_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.3_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.5_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.5_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.2_last_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.2_last_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp1L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp1L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstm2L_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstm3L_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstm2L_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstm3L_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'resume_lstm2L_4+5_strategy=post_4+7_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstm2L_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_9.pth',
    # 'resume_lstm2L_4+5_strategy=post_4+7_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstm2L_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_10.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_avgpool_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_avgpool_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.2_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.3_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.2_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.3_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_layerdecay=0.6_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_layerdecay=0.8_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_layerdecay=1.2_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_layerdecay=1.4_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_layerdecay=0.6_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_layerdecay=0.8_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_layerdecay=1.2_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_layerdecay=1.4_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_lstmdrp=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_descending_interval=2_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_descending_interval=2_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_ascending_interval=2_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'fix_index_strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstmln_mlpdrp=0.4_layerwise_lstmdrp=0.3_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_jointlr_0.000067_exitscale=0.25_7.pth',
    # 'strategy=post_4+5_exit_layer_11_multi-exit_ascending_interval=2_extra-exit_mlp2L_mlpln_lstmln_lr_scale=0.25_mlpdrp=0.4_layerwise_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    # 'fix_index_strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlp2L_mlpln_lstmln_mlpdrp=0.4_layerwise_lstmdrp=0.3_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_jointlr_0.000067_exitscale=0.25_7.pth',
    # 'fix_index_strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_nodetach_mlp2L_mlpln_lstmln_mlpdrp=0.4_layerwise_lstmdrp=0.3_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_jointlr_0.000067_exitscale=0.25_7.pth',
    # 'stg=post_4+5_layer_23_multie_intv=2_extrae_nodth_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'stg=post_4+5_layer_11_multie_intv=2_extrae_nodth_lwproj1L_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'stg=post_4+5_layer_11_multie_intv=2_extrae_nodth_lwproj2L_res_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'stg=post_4+5_layer_11_multie_intv=2_extrae_nodth_reg_lwproj1L_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    # 'stg=post_4+5_layer_11_multie_intv=2_extrae_nodth_reg_lwproj2L_res_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    
    # D
    # 'stg=post_4+5_layer_11_multie_intv=2_extrae_nodth_reg_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    'stg=post_4+5_layer_11_multie_noloss_intv=2_extrae_nodth_reg_mlpdrp=0.4_layerwise_lstmdrp=0.3_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    'stg=post_4+5_layer_11_multie_noloss_intv=2_extrae_nodth_reg_mlpdrp=0.4_layerwise_lstmdrp=0.3_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_freeze_emb_7.pth',

    # D L24
    'stg=post_4+5_layer_23_multie_intv=2_extrae_nodth_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    
    # original flamingo
    # 'checkpoint_gripper_post_hist_1__exit_layer_-1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth4.pth', 
    # 'checkpoint_gripper_post_hist_1__exit_layer_11_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth4.pth', 
    # 'checkpoint_gripper_post_hist_1__exit_layer_5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth4.pth',


    # ABCD
    # 'stg=post_3+3_layer_11_multie_intv=2_extrae_nodth_reg_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth',
    # 'stg=post_3+3_layer_11_multie_intv=2_extrae_nodth_reg_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth',
    # 'stg=post_3+1_layer_11_multie_intv=2_extrae_nodth_reg_mlpdrp=0.5_layerwise_lstmdrp=0.4_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth',
    # 'stg=post_3+1_layer_11_multie_intv=2_extrae_nodth_reg_mlpdrp=0.5_layerwise_lstmdrp=0.4_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_60000_iter.pth',
    # 'stg=post_3+1_layer_11_multie_intv=2_extrae_nodth_reg_mlpdrp=0.5_layerwise_lstmdrp=0.4_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_57500_iter.pth',
    # 'stg=post_3+1_layer_11_multie_intv=2_extrae_nodth_reg_mlpdrp=0.5_layerwise_lstmdrp=0.4_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_55000_iter.pth',
    # use tstate (seemd wrong and indeed don't use state)
    # 'stg=post_3+0_layer_11_multie_intv=2_extrae_nodth_reg_mlpdrp=0.4_layerwise_lstmdrp=0.3_state_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth',
    # no multiexit auxiliary loss
    'stg=post_3+0_layer_23_multie_noloss_intv=2_extrae_nodth_reg_mlpdrp=0.4_layerwise_lstmdrp=0.3_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth',

    
    # ABC
    'stg=post_4+4_layer_11_multie_intv=2_extrae_nodth_reg_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth',
    
    # 9B
    'stg=post_4+4_layer_15_multie_intv=2_extrae_nodth_reg_mlpdrp=0.4_layerwise_lstmdrp=0.3_aug_10_4_traj_cons_ws_12_mpt_9b_7.pth',
]


print(ckpt_names)
for ckpt_name in ckpt_names:
    for r in args.exit_ratio:
        # use_gripper = 1 if 'gripper' in ckpt_name else 0
        # use_state = 1 if 'state' in ckpt_name else 0
        use_gripper = 1
        use_state = 0
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print("ckpt doesn't exist, skipped.")
            continue
        if args.value_net_ckpt_dir:
            value_net_ckpt_path = os.path.join(args.value_net_ckpt_dir, ckpt_name[:-4]+'_value_net_mlp_b20_1.pth')
        else:
            value_net_ckpt_path = 'None'
        log_dir = f'log_{args.ckpt_dir}'
        os.makedirs(log_dir, exist_ok=True)
        prefix = f'evaluate{args.num_seq}{args.note}_{args.precision}'
        if args.enrich_annotation:
            prefix += '_enrich'
        if args.layerwise_exit_eval:
            prefix += '_per_exit'
        if args.amp:
            prefix += '_amp'
        prefix += f'_{args.eval_exit_mode}'
        if args.eval_exit_mode == 'dynamic':
            prefix += f'_{args.value_type}'
            if args.value_type == 'action':
                prefix += f'_{args.threshold_type}'
                if args.use_action_ensemble:
                    prefix += f'_ensemble'
            prefix += f'_maxL={args.max_layer}_{args.exit_dist}_{r}'
        prefix += '_exit'
        if args.multi_execution > 1:
            prefix += f'_{args.multi_execution}_execution'
            
        if args.eval_exit_mode != 'dynamic' or args.value_type != 'loss':
            # log_file = '{}/{}_{}.log'.format(log_dir, prefix, '.'.join(ckpt_name.split('.')[:-1]))
            iter = ckpt_name[-5] if 'iter' not in ckpt_name else ckpt_name[-14:-9]
            log_file = '{}/{}_{}.log'.format(log_dir, prefix, ckpt_name[:-30]+iter)
        else:
            log_file = '{}/{}_{}.log'.format(log_dir, prefix, os.path.basename(value_net_ckpt_path)[:30]+os.path.basename(value_net_ckpt_path)[-2:])
        if 'freeze_emb' in ckpt_name:
            log_file = log_file[:-4] + '_freeze_emb.log'
        if os.path.exists(log_file): 
            print(f'skip {log_file}')
            continue
        ckpt_ix = ckpt_names.index(ckpt_name)
        print('evaluating {}/{} checkpoint'.format(ckpt_ix+1, len(ckpt_names)))
        # fusion_mode = 'pre'
        # if 'post' in ckpt_name:
        #     fusion_mode = 'post'
        # if 'two_way' in ckpt_name:
        #     fusion_mode = 'two_way'
        fusion_mode = 'post'
        window_size = 8
        ckpt_attrs = ckpt_name.split('_')
        if 'ws' in ckpt_attrs:
            window_size = int(ckpt_attrs[ckpt_attrs.index('ws')+1])
        # print('bash robot_flamingo/pt_eval_ckpts.bash {} {} {} {}'.format(ckpt_path, log_file, use_gripper, use_state))
        # exit(0)

        os.system('bash robot_flamingo/pt_eval_ckpts.bash {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(ckpt_path, log_file, use_gripper, 
            use_state, fusion_mode, window_size, args.node_num, args.single_step, args.amp, args.eval_exit_mode, args.multi_execution, value_net_ckpt_path, r, 
            args.layerwise_exit_eval, args.value_type, args.num_seq, args.threshold_type, args.use_action_ensemble, args.exit_dist, args.max_layer, args.enrich_annotation, args.precision))

