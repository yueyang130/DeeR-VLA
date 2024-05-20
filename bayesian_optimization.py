
import argparse
import os
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args



parser = argparse.ArgumentParser()
# bayesian optimization
parser.add_argument('--num_seq', type=int)
parser.add_argument("--evaluate_from_checkpoint", type=str)
parser.add_argument("--acq_func", type=str, default='EI', choices=['EI', 'LCB', 'PI'])
parser.add_argument('--n_calls', type=int)
parser.add_argument('--init_exit_ratio', type=float)
args = parser.parse_args()


ckpt_dir, ckpt_name = os.path.split(args.evaluate_from_checkpoint)
log_file = 'log_BO/' +  ckpt_dir + ckpt_name[:-4] + '.log'
iter_num = 0
print(f'{log_file=}')

# solve thresholds with exp distribution with a validation datast
# os.system(f"""
#     torchrun --nnodes=1 --nproc_per_node=$ARNOLD_WORKER_GPU  --master_port=$METIS_WORKER_0_PORT robot_flamingo/eval/eval_calvin.py \
#     --precision fp32 \
#     --use_gripper \
#     --window_size 12 \
#     --fusion_mode post \
#     --run_name RobotFlamingoDBG \
#     --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
#     --cross_attn_every_n_layers 4 \
#     --evaluate_from_checkpoint {args.evaluate_from_checkpoint} \
#     --calvin_conf_path /mnt/bn/yueyang/archive/calvin/calvin_models/conf \
#     --amp true \
#     --eval_exit_mode dynamic \
#     --exit_ratio {args.init_exit_ratio} \
#     --value_type action \
#     --threshold_type L2 --exit_dist exp \
#     --num_seq {args.num_seq} \
#     --validation_set \
#     --workers 1 > {log_file} 2>&1
# """)
    
with open(log_file, 'r') as file:
    lines = file.readlines()
    thresholds_str = lines[-3]
    init_thresholds = list(map(float, thresholds_str.split(',')))
    init_avg_len = float(lines[-2])  
    init_avg_exit = budget = float(lines[-1])
    
print('exp result:')
print(init_thresholds)
print(init_avg_len)
print(init_avg_exit)


space = [
    Real(init_thresholds[0]-0.02, init_thresholds[0]+0.02, name='t0'),
    Real(init_thresholds[1]-0.002, init_thresholds[1]+0.002, name='t1'),
    Real(init_thresholds[2]-0.002, init_thresholds[2]+0.002, name='t2'),
    Real(init_thresholds[3]-0.002, init_thresholds[3]+0.002, name='t3'),
    Real(init_thresholds[4]-0.002, init_thresholds[4]+0.002, name='t4'),
]

@use_named_args(space)
def objective_function(t0, t1, t2, t3, t4):
    global log_file
    global iter_num
    iter_num += 1
    log_file  = log_file[:-4] + f'_iter{str(iter_num)}' + '.log'
    t5 = 100000.0
    if not os.path.exists(log_file):
        os.system(f"""
        torchrun --nnodes=1 --nproc_per_node=$ARNOLD_WORKER_GPU  --master_port=$METIS_WORKER_0_PORT robot_flamingo/eval/eval_calvin.py \
        --precision fp32 \
        --use_gripper \
        --window_size 12 \
        --fusion_mode post \
        --run_name RobotFlamingoDBG \
        --calvin_dataset /mnt/bn/yueyang/archive/calvin/dataset/task_D_D \
        --cross_attn_every_n_layers 4 \
        --evaluate_from_checkpoint {args.evaluate_from_checkpoint} \
        --calvin_conf_path /mnt/bn/yueyang/archive/calvin/calvin_models/conf \
        --amp true \
        --eval_exit_mode dynamic \
        --thresholds {t0} {t1} {t2} {t3} {t4} {t5} \
        --value_type action \
        --threshold_type L2 --exit_dist exp \
        --num_seq {args.num_seq} \
        --validation_set \
        --workers 1 > {log_file} 2>&1
        """)
    
    with open(log_file, 'r') as file:
        lines = file.readlines()
        thresholds_str = lines[-3]
        thresholds = list(map(float, thresholds_str.split(',')))
        avg_len = float(lines[-2])  
        avg_exit = float(lines[-1])
        
    print('')
    print(f'{iter_num=}')
    print(f'{thresholds=}')
    print(f'{avg_len=}')
    print(f'{avg_exit=}')
    
    if avg_exit > budget:
        res =  0
    else:
        res = - avg_len
    print(f'BO {res=}')
    return res


result = gp_minimize(
    objective_function, 
    space, 
    x0=init_thresholds[:-1], 
    y0=init_avg_len, 
    n_calls=20, 
    random_state=0
)

print("Optimal thresholds:", result.x)
print("optimal avg exit:", -result.fun)