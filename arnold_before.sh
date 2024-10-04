#!/bin/bash
# This script is for 1) Install dependancies; 2) Align internal cluster with standard practice

# set port
# if env SSH_LOGIN_KEYS is set, no need for ssh generation
# cd ~/.ssh
# ssh-keygen -t rsa -b 4096 -f ssh_host_rsa_key -N '' # press Enter all the time here
# # mkdir /run/sshd # NOTE: only cluster name "lab_wj" needs this step.
# /usr/sbin/sshd -h ~/.ssh/ssh_host_rsa_key -p $METIS_WORKER_0_PORT # METIS_WORKER_0_PORT set in ENV VAR
# hostname -I | awk '{split($0, a, " "); print a[1]}'
# echo $METIS_WORKER_0_PORT # get your port number here, such as 9000

# Pip install 
# pip3 install --upgrade pip
# sudo pip3 uninstall -y torch torchaudio torchvision # reinstall torch 2.0
# pip3 install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118

export http_proxy=http://sys-proxy-rd-relay.byted.org:3128 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org
echo "export http_proxy=http://sys-proxy-rd-relay.byted.org:3128 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org" >> ~/.bashrc
export WANDB_API_KEY=3e0863e2d8f819730b85529bd24b3ebbb96d0eb3

pip3 install 'setuptools<58.0.0'
pip3 install pyhash
pip3 install GPUtil
pip3 install transformers
pip install wandb_osh

HOME_PATH=/mnt/bn/yueyang

# add conda to PATH
echo "export PATH=/mnt/bn/yueyang/miniconda/bin:$PATH" >> ~/.bashrc
# conda create -n RoboFlamingo python=3.10.6
# source activate RoboFlamingo
# pip install -r requirements.txt # already installed in mount

# sudo apt-get install rustc -y

# install calvin
cd $HOME_PATH/archive/calvin
pip3 install wheel cmake==3.18.4
cd calvin_env/tacto
python3 setup.py egg_info
pip3 install -e .
cd ..
pip3 install -e .
cd ../calvin_models
python3 setup.py egg_info
pip3 install -e .

# reinstall torchaudio
pip uninstall torch torchaudio
# sudo pip uninstall pytorch-lightning -y
pip install torchaudio==0.13.1
# pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
pip3 install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install optimum-quanto

# install jax and cudn, cudnn
sudo apt install ffmpeg -y

# for the error "failed to EGL with glad."
sudo apt-get install libglfw3-dev libgles2-mesa-dev -y

# for the error "ImportError: cannot import name 'gcd' from 'fractions' (/usr/lib/python3.9/fractions.py)"
pip install --upgrade networkx


sudo apt install iputils-ping gpustat -y
sudo apt install kmod

# install roboflamingo requirements
sudo apt-get install gfortran libopenblas-dev liblapack-dev -y
sudo apt-get install python3-tk -y
sudo apt-get install python3-scipy -y
cd $HOME_PATH/RoboFlamingo
pip3 install -r requirements.txt

# Install dependency for calvin
sudo apt-get -y install libegl1-mesa libegl1
sudo apt-get -y install libgl1

sudo apt-get update -y -qq
sudo apt-get install -y -qq libegl1-mesa libegl1-mesa-dev

sudo apt install -y mesa-utils libosmesa6-dev llvm
sudo apt-get -y install meson
sudo apt-get -y build-dep mesa

# cd $HOME_PATH/archive/pytorch3d
# pip install -e .

echo "export PYTHONPATH=$PYTHONPATH:/mnt/bn/yueyang/RoboFlamingo/robot_flamingo:/mnt/bn/yueyang/RoboFlamingo/open_flamingo" >> ~/.bashrc
echo "export WANDB_API_KEY=3e0863e2d8f819730b85529bd24b3ebbb96d0eb3" >> ~/.bashrc
export PYTHONPATH=/mnt/bn/yueyang/RoboFlamingo/robot_flamingo:/mnt/bn/yueyang/RoboFlamingo/open_flamingo:$PYTHONPATH
export WANDB_API_KEY=3e0863e2d8f819730b85529bd24b3ebbb96d0eb3

source ~/.bashrc

cd $HOME_PATH/RoboFlamingo

# for the error: module 'numpy' has no attribute 'float'
pip install 'numpy==1.23'
pip install thop fvcore
pip install scikit-optimize
pip install bitsandbytes accelerate

# SimplerEnv
echo "export XDG_RUNTIME_DIR=/run/user/$(id -u)" >> ~/.bashrc
sudo apt-get install libvulkan1
sudo apt-get install vulkan-tools
sudo apt-get install libglvnd-dev

mkdir -p /usr/share/vulkan/icd.d
cd /usr/share/vulkan/icd.d
sudo wget  https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/nvidia_icd.json
sudo wget -q -O /usr/share/glvnd/egl_vendor.d/10_nvidia.json https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/10_nvidia.json
# cd /mnt/bn/yueyang/SimplerEnv
# cd ManiSkill2_real2sim/
# pip install -e .
# cd ..
# pip install -e .
# pip install --upgrade mani_skill
# sudo apt install  libnvidia-rtcore libnvoptix1 libnvidia-ngx1 
# cd ../DeeR_real/
# sudo cp vulkaninfo/10_nvidia.json /usr/share/glvnd/egl_vendor.d/
# sudo cp vulkaninfo/nvidia_icd.json /usr/share/vulkan/icd.d/
# sudo cp vulkaninfo/nvidia_icd.json /etc/vulkan/icd.d/
# # sudo cp vulkaninfo/nvidia_icd.json /usr/share/icd.d/
# sudo cp vulkaninfo/nvidia_layers.json /etc/vulkan/implicit_layer.d/


# unset http_proxy && unset https_proxy && unset no_proxy

 ssh-keygen
 chmod 600 /home/tiger/.ssh/id_rsa

 cd /mnt/bn/yueyang/RoboFlamingo
 

# ----------------------------------------------------------------------------------------
# setup environment variables
# disable TF verbose logging
TF_CPP_MIN_LOG_LEVEL=2
# fix known issues for pytorch-1.5.1 accroding to 
# https://blog.exxactcorp.com/pytorch-1-5-1-bug-fix-release/
MKL_THREADING_LAYER=GNU
# set NCCL envs for disributed communication
NCCL_IB_GID_INDEX=3
NCCL_IB_DISABLE=0
NCCL_DEBUG=INFO
ARNOLD_FRAMEWORK=pytorch

# get distributed training parameters 
METIS_WORKER_0_HOST={METIS_WORKER_0_HOST:127.0.0.1}
NV_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
ARNOLD_WORKER_GPU={ARNOLD_WORKER_GPU:-$NV_GPUS}
ARNOLD_WORKER_NUM={ARNOLD_WORKER_NUM:-1}
ARNOLD_ID={ARNOLD_ID:-0}

NNODES=$ARNOLD_WORKER_NUM
NODE_RANK=$ARNOLD_ID
MASTER_ADDR=$METIS_WORKER_0_HOST
GPUS=$ARNOLD_WORKER_GPU
# generate a random port from the ranage (20000, 40000)
RAND_PORT=$(( ((RANDOM<<15)|RANDOM) % 20000 + 20000 ))
# PORT={PORT:-$RAND_PORT}
PORT=3343
