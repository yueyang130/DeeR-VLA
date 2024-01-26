import os
import glob
import argparse

# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
parser.add_argument('--ckpt_dir', type=str, help='The checkpoint directory')
parser.add_argument('--single_step', action='store_true', help='If set, evlauate in single step mode')
parser.add_argument('--node_num', type=int)

# Parse the arguments
args = parser.parse_args()

search_path = os.path.join(args.ckpt_dir,  '*gaussian*.pth')
ckpt_names = [os.path.basename(path) for path in glob.glob(search_path)]
ckpt_names.sort(reverse=True)
# ckpt_names = ckpt_names[-1:]
# ckpt_names = [
    # 'checkpoint_gripper_post_hist_1__exit_layer_-1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_3.pth3.pth',
    # 'checkpoint_gripper_post_hist_1__exit_layer_-1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_4.pth4.pth',
# ]

print(ckpt_names)
for ckpt_name in ckpt_names:
    use_gripper = 1 if 'gripper' in ckpt_name else 0
    use_state = 1 if 'state' in ckpt_name else 0
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
    log_dir = f'log_{args.ckpt_dir}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = '{}/evaluate_{}.log'.format(log_dir, '.'.join(ckpt_name.split('.')[:-1]))
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
    os.system('bash robot_flamingo/pt_eval_ckpts.bash {} {} {} {} {} {} {} {}'.format(ckpt_path, log_file, use_gripper, use_state, fusion_mode, window_size, args.node_num, args.single_step))

