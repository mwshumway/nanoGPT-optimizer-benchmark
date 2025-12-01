#!/bin/bash -l

# Set SCC project
#$ -P replearn

# Request 8 cores
#$ -pe omp 8

# Request 3 gpus
#$ -l gpus=1

# Minimum compute capability
#$ -l gpu_c=8.0

# Runtime
#$ -l h_rt=24:00:00

module load miniconda
conda activate modded-nanogpt
module load cuda/12.5

python train.py config/train_shakespeare_char.py \
--optimizer_variant=full_matrix_adagrad \
--device='cuda' \
--compile=True \
--eval_iters=20 \
--log_interval=1 \
--block_size=32 \
--batch_size=32 \
--n_layer=4 \
--n_head=4 \
--n_embd=32 \
--max_iters=2000 \
--lr_decay_iters=2000 \
--dropout=0.0 \
--wandb_log=True \
--wandb_group_name='full-matrix-adagrad' \
--wandb_run_name='full-matrix-adagrad-lr0.1' \
--wandb_project='shakespeare-char-small' \
--eval_interval=100 \
--lr_finder=False \
--learning_rate=0.1 \
--min_lr=0.01 \


