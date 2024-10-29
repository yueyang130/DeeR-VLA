import os
import glob
import argparse

# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
parser.add_argument('--ckpt_dir', type=str, help='The checkpoint directory')
parser.add_argument('--exit_ratio', nargs='+', type=float, default=[1.0], help='a list')
parser.add_argument('--node_num', type=int, help='how much GPUs/threads to parallelly evaluate')
parser.add_argument("--num_seq", type=int, default=224, help="the number of task chains for elvaution. Maximum is 1000.")
parser.add_argument(
        "--amp",
        default=0,
        type=int,
    )
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

search_path = os.path.join(args.ckpt_dir,  r'*_[0-9].pth')
ckpt_names = [os.path.basename(path) for path in glob.glob(search_path)]
ckpt_names.sort(reverse=True)

print(ckpt_names)
for ckpt_name in ckpt_names:
    for r in args.exit_ratio:
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print("ckpt doesn't exist, skipped.")
            continue
        log_dir = f'log_{args.ckpt_dir}'
        os.makedirs(log_dir, exist_ok=True)
        prefix = f'evaluate{args.num_seq}{args.note}_{args.precision}'
        if args.enrich_annotation:
            prefix += '_enrich'
        if args.amp:
            prefix += '_amp'
        prefix += f'_maxL={args.max_layer}_{r}'
        prefix += '_exit'
            
        log_file = '{}/{}_{}.log'.format(log_dir, prefix, '.'.join(ckpt_name.split('.')[:-1]))

        if 'freeze_emb' in ckpt_name:
            log_file = log_file[:-4] + '_freeze_emb.log'
        if os.path.exists(log_file): 
            print(f'skip {log_file}')
            continue
        ckpt_ix = ckpt_names.index(ckpt_name)
        print('evaluating {}/{} checkpoint'.format(ckpt_ix+1, len(ckpt_names)))

        window_size = 12
        ckpt_attrs = ckpt_name.split('_')
        if 'ws' in ckpt_attrs:
            window_size = int(ckpt_attrs[ckpt_attrs.index('ws')+1])

        os.system('bash robot_flamingo/pt_eval_ckpts.bash {} {} {} {} {} {} {} {} {} {}'.format(
            ckpt_path,
            log_file,
            window_size,
            args.node_num,
            args.amp,
            r, 
            args.num_seq,
            args.max_layer,
            args.enrich_annotation,
            args.precision))