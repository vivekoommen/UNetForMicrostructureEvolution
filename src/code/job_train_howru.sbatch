#!/bin/bash
#SBATCH --time=990:00:00
#SBATCH --mem=512g
#SBATCH -N 1
#SBATCH -n 8
##SBATCH --ntasks-per-node=8
#SBATCH --partition=3090-gcondo
##SBATCH --partition=a6000-gcondo
#SBATCH --nodelist=gpu2101,gpu2102,gpu2103,gpu2104,gpu2105,gpu2106,gpu2107,gpu2108,gpu2109
#SBATCH --gres=gpu:1

#SBATCH -J train_howru
#SBATCH -o train_howru.out
#SBATCH -e train_howru.err
## SBATCH --exclusive

module load cuda

source ~/apps/torch_env/bin/activate

XLA_FLAGS=--xla_gpu_cuda_data_dir=/oscar/rt/9.2/software/0.20-generic/0.20.1/opt/spack/linux-rhel9-x86_64_v3/gcc-11.3.1/cuda-12.2.0-4lgnkrhcomlf3rt3mirnu7wtmgyxtrbd python3 -u train_howru.py



