
python3.10 train.py config/train_shakespeare_char.py \
--optimizer_variant=full_matrix_adagrad \
--device='mps' \
--compile=False \
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
--wandb_run_name='full-matrix-adagrad-lr1.0' \
--wandb_project='shakespeare-char-small' \
--eval_interval=100 \
--lr_finder=False \
--learning_rate=1.0 \
--min_lr=0.1 \


