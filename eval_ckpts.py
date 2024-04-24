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
parser.add_argument(
        "--amp",
        default=False,
        action="store_true"
    )
parser.add_argument("--eval_exit_mode", type=str, default='last', choices=['last', 'all', 'dynamic']) # only eval the last exit / all exits / dynamic early-exit mechanism
parser.add_argument("--multi_execution", type=int, default=1, help="how many actions are executed in one time when predicting multiple actions; if only one predicted action, repeat it K times")

# Parse the arguments
args = parser.parse_args()

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
    'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth',
    'strategy=post_4+5_exit_layer_11_multi-exit_uniform_interval=2_extra-exit_mlpln_lstmln_lr_scale=0.25_dropout=0.1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_8.pth',
    
]

print(ckpt_names)
for ckpt_name in ckpt_names:
    for r in args.exit_ratio:
        # use_gripper = 1 if 'gripper' in ckpt_name else 0
        # use_state = 1 if 'state' in ckpt_name else 0
        use_gripper = 1
        use_state = 0
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
        # value_net_ckpt_path = os.path.join(args.value_net_ckpt_dir, ckpt_name[:-4]+'_value_net_4.pth')
        value_net_ckpt_path = os.path.join(args.value_net_ckpt_dir, ckpt_name[:-4]+'_value_net_discrete_b20_4.pth')
        log_dir = f'log_{args.ckpt_dir}'
        os.makedirs(log_dir, exist_ok=True)
        prefix = 'evaluate'
        if args.amp:
            prefix += '_amp'
        prefix += f'_{args.eval_exit_mode}'
        if args.eval_exit_mode == 'dynamic':
            print(f'eval exit ratio = {r}')
            prefix += f'_{r}'
        prefix += '_exit'
        if args.multi_execution > 1:
            prefix += f'_{args.multi_execution}_execution'
            
        if args.eval_exit_mode != 'dynamic':
            log_file = '{}/{}_{}.log'.format(log_dir, prefix, '.'.join(ckpt_name.split('.')[:-1]))
        else:
            log_file = '{}/{}_{}.log'.format(log_dir, prefix, '.'.join(os.path.basename(value_net_ckpt_path).split('.')[:-1]))
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

        os.system('bash robot_flamingo/pt_eval_ckpts.bash {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(ckpt_path, log_file, use_gripper, 
            use_state, fusion_mode, window_size, args.node_num, args.single_step, args.amp, args.eval_exit_mode, args.multi_execution, value_net_ckpt_path, r))

