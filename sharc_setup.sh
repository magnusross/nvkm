qrshx  -l rmem=16G -l gpu=1
qrshx -l rmem=32G -l gpu=1 -P rse -q rse-interactive.q
module load apps/python/conda
source activate nvkm
module load libs/cudnn/8.0.5.39/binary-cuda-11.1.1
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/packages/libs/CUDA/11.1.1/binary/
export JAX_ENABLE_X64=True