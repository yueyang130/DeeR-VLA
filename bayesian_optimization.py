
import argparse
import os
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args


def get_observation(log_path):
    with open(log_path, 'r') as file:
        lines = file.readlines()
        thresholds_str = lines[-3]
        thresholds = list(map(float, thresholds_str.split(',')))
        avg_len = float(lines[-2])  
        avg_exit = float(lines[-1])
    return thresholds, avg_len, avg_exit

def get_score(avg_len, avg_exit):
    if avg_exit > budget:
        res =  - avg_len + 1.0 * (avg_exit - budget)
    else:
        res = - avg_len
    return res

parser = argparse.ArgumentParser()
# bayesian optimization
parser.add_argument('--num_seq', type=int)
parser.add_argument("--evaluate_from_checkpoint", type=str)
parser.add_argument("--acq_func", type=str, default='EI', choices=['EI', 'LCB', 'PI'])
parser.add_argument('--n_calls', type=int)
parser.add_argument('--init_exit_ratio', type=float)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--port', type=int)
args = parser.parse_args()

assert os.environ['calvin_dataset_path'] and os.environ['calvin_conf_path'], "PLEASE SET CAVLIN DATASET PATH and CONFIG PATH!"
args.calvin_dataset = os.environ['calvin_dataset_path']
args.calvin_conf_path = os.environ['calvin_conf_path']


ckpt_dir, ckpt_name = os.path.split(args.evaluate_from_checkpoint)
log_dir = f'log_BO_{args.init_exit_ratio}_{ckpt_dir}/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


iter_num = 0
log_file = log_dir + f'seq{args.num_seq}_{args.acq_func}_seed{args.seed}_' + ckpt_name[:-4] + f'_iter{str(iter_num)}' + '.log'
print(f'{log_file=}')

# solve thresholds with exp distribution with a validation datast to get initial point for bayesian optimization
if not os.path.exists(log_file):
    os.system(f"""
        torchrun --nnodes=1 --nproc_per_node=$ARNOLD_WORKER_GPU  --master_port={args.port} robot_flamingo/eval/eval_calvin.py \
        --precision fp32 \
        --use_gripper \
        --run_name DeeR \
        --calvin_dataset {args.calvin_dataset} \
        --cross_attn_every_n_layers 4 \
        --evaluate_from_checkpoint {args.evaluate_from_checkpoint} \
        --calvin_conf_path {args.calvin_conf_path} \
        --amp 1 \
        --exit_ratio {args.init_exit_ratio} \
        --num_seq {args.num_seq} \
        --validation_set \
        --workers 1 > {log_file} 2>&1
    """)
    
with open(log_file, 'r') as file:
    lines = file.readlines()
    thresholds_str = lines[-3]
    init_thresholds = list(map(float, thresholds_str.split(',')))
    init_avg_len = float(lines[-2])  
    # set the FLOPs of the running using demonstration dataset as budget constraint,
    # such that the search result by bayesian should cost less FLOPs than threshold only using demonstration dataset.
    # You can set other values manually. PLEASE that here all values represents the average exit layer.  
    # Average exit layer * FLOPS per layer = Average FLOPs
    init_avg_exit = budget = float(lines[-1]) 
    
print('exp result:')
print(init_thresholds)
print(init_avg_len)
print(init_avg_exit)


# get existing observations as other initial points
x0, y0 = [init_thresholds[:-1]], [-init_avg_len]
from pathlib import Path
for log in Path(log_dir).glob('*.log'):
    if 'iter0.log' in str(log): continue
    try:
        thresholds, avg_len, avg_exit = get_observation(log)
        score = get_score(avg_len, avg_exit)     
        x0.append(thresholds[:-1])
        y0.append(score)
    except:
        print(f'Error when parsing {log}')
        pass
    
# define search space
space = [
    Real(init_thresholds[0]-0.02, init_thresholds[0]+0.02, name='t0'),
    Real(init_thresholds[1]-0.002, init_thresholds[1]+0.002, name='t1'),
    Real(init_thresholds[2]-0.002, init_thresholds[2]+0.002, name='t2'),
    Real(init_thresholds[3]-0.002, init_thresholds[3]+0.002, name='t3'),
    Real(init_thresholds[4]-0.002, init_thresholds[4]+0.002, name='t4'),
]
# space = [
#     Real(init_thresholds[0]-0.01, init_thresholds[0]+0.01, name='t0'),
#     Real(init_thresholds[1]-0.001, init_thresholds[1]+0.001, name='t1'),
#     Real(init_thresholds[2]-0.001, init_thresholds[2]+0.001, name='t2'),
#     Real(init_thresholds[3]-0.001, init_thresholds[3]+0.001, name='t3'),
#     Real(init_thresholds[4]-0.001, init_thresholds[4]+0.001, name='t4'),
# ]

@use_named_args(space)
def objective_function(t0, t1, t2, t3, t4):
    global log_file
    global iter_num
    iter_num += 1
    log_file  = log_file[:-10] + f'_iter{str(iter_num)}' + '.log'
    t5 = 100000.0
    
    print('')
    print(f'{iter_num=}')
    print(f'threshold={t0}, {t1}, {t2}, {t3}, {t4}, {t5}')
    
    if not os.path.exists(log_file):
        os.system(f"""
        torchrun --nnodes=1 --nproc_per_node=$ARNOLD_WORKER_GPU  --master_port={args.port} robot_flamingo/eval/eval_calvin.py \
        --precision fp32 \
        --use_gripper \
        --run_name DeeR \
        --calvin_dataset {args.calvin_dataset} \
        --cross_attn_every_n_layers 4 \
        --evaluate_from_checkpoint {args.evaluate_from_checkpoint} \
        --calvin_conf_path {args.calvin_conf_path} \
        --amp 1 \
        --thresholds {t0} {t1} {t2} {t3} {t4} {t5} \
        --num_seq {args.num_seq} \
        --validation_set \
        --workers 1 > {log_file} 2>&1
        """)
    
    thresholds, avg_len, avg_exit = get_observation(log_file)
    res = get_score(avg_len, avg_exit)
    print(f'{avg_len=}')
    print(f'{avg_exit=}')
    print(f'BO {res=}')
    return res

# print('')
# print('init x0:', x0) 
print('init y0:', y0) 

result = gp_minimize(
    objective_function, 
    space, 
    x0=x0, 
    y0=y0, 
    n_calls=20, 
    random_state=args.seed,
    acq_func=args.acq_func,  # 选择采集函数
)

print("Optimal thresholds:", result.x)
print("optimal avg exit:", -result.fun)