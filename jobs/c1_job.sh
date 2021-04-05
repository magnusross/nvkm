#!/bin/bash
#$ -l gpu=1
#$ -P rse
#$ -q rse.q
#$ -M mross1@sheffield.ac.uk
#$ -m a
#$ -l rmem=16G
#$ -l h_rt=03:00:00


module load apps/python/conda
source activate nvkm
module load libs/cudnn/8.0.5.39/binary-cuda-11.1.1
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/packages/libs/CUDA/11.1.1/binary/
python ../c1_experiment.py --Nvu 88 --Nvg 20  --Nits 1000 --lr 1e-2 --Nbatch 100 --Ns 50 --ampsgs_init 1.5 --lsgs 0.8 --fit_noise 0 --f_name c1_t_

