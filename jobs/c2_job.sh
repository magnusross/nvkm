#!/bin/bash
#$ -l gpu=1
#$ -P rse
#$ -q rse.q
#$ -M mross1@sheffield.ac.uk
#$ -m a
#$ -l rmem=12G
#$ -l h_rt=01:00:00


module load apps/python/conda
source activate nvkm
module load libs/cudnn/8.0.5.39/binary-cuda-11.1.1
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/packages/libs/CUDA/11.1.1/binary/
python ../c2_experiment.py --Nvu 20 --Nvg 10 --Ndata 200 --Nits 1000 --lr 1e-3 --Nbatch 50 --Ns 30 --q_frac 0.5 --ampsgs_init 1.0 0.1 --lsgs 1.0 1.0 --fit_noise 0 --alpha 0.4 --f_name c2_t
