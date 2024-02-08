import os
import glob
import argparse

# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
parser.add_argument('--ckpt_dir', type=str, help='The checkpoint directory')
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
    'amp_checkpoint_gripper_post_hist_1__exit_layer_5_data_0.5_aug_10_4_3_step_traj_cons_ws_12_mpt_dolly_3b_4.pth',
    
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth',
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_lr_scale=0.25_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth',
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_lr_scale=0.25_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth',
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_lr_scale=0.25_dropout=0.1_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth',
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_lr_scale=0.25_dropout=0.2_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth',
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_lr_scale=0.25_dropout=0.2_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth',
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_lr_scale=0.25_dropout=0.5_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth',
    # 'amp_checkpoint_gripper_post_hist_1__exit_layer_5_multi-exit_uniform_interval=1_lr_scale=0.25_dropout=0.5_data_0.5_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth',
]

print(ckpt_names)
for ckpt_name in ckpt_names:
    use_gripper = 1 if 'gripper' in ckpt_name else 0
    use_state = 1 if 'state' in ckpt_name else 0
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
    log_dir = f'log_{args.ckpt_dir}'
    os.makedirs(log_dir, exist_ok=True)
    prefix = 'evaluate'
    if args.amp:
        prefix += '_amp'
    prefix += f'_{args.eval_exit_mode}_exit'
    prefix += f'_{args.multi_execution}_execution'
        
    log_file = '{}/{}_{}.log'.format(log_dir, prefix, '.'.join(ckpt_name.split('.')[:-1]))
    if os.path.exists(log_file): 
        print(f'skip {ckpt_name}')
        continue
    ckpt_ix = ckpt_names.index(ckpt_name)
    print('evaluating {}/{} checkpoint'.format(ckpt_ix+1, len(ckpt_names)))
    fusion_mode = 'pre'
    if 'post' in ckpt_name:
        fusion_mode = 'post'
    if 'two_way' in ckpt_name:
        fusion_mode = 'two_way'
    window_size = 8
    ckpt_attrs = ckpt_name.split('_')
    if 'ws' in ckpt_attrs:
        window_size = int(ckpt_attrs[ckpt_attrs.index('ws')+1])
    # print('bash robot_flamingo/pt_eval_ckpts.bash {} {} {} {}'.format(ckpt_path, log_file, use_gripper, use_state))
    # exit(0)
    os.system('bash robot_flamingo/pt_eval_ckpts.bash {} {} {} {} {} {} {} {} {} {} {}'.format(ckpt_path, log_file, use_gripper, 
        use_state, fusion_mode, window_size, args.node_num, args.single_step, args.amp, args.eval_exit_mode, args.multi_execution))

