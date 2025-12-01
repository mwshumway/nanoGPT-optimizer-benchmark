export PYTORCH_ENABLE_MPS_FALLBACK=1
python train.py config/train_shakespeare_char.py \
--optimizer_variant=full_matrix_adagrad \
--device='mps' \
--compile=False \
--eval_iters=20 \
--log_interval=1 \
--block_size=16 \
--batch_size=64 \
--n_layer=4 \
--n_head=4 \   
--n_embd=32 \ 
--max_iters=100 \ 
--lr_decay_iters=2000 \
--dropout=0.0 \
--wandb_log=True \
--wandb_group_name='full_matrix_adagrad' \
--wandb_run_name='full_matrix_adagrad-base' 
