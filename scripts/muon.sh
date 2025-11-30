
python3.10 train.py config/train_shakespeare_char.py \
--optimizer_variant=muon \
--device='mps' \
--compile=False \
--eval_iters=20 \
--log_interval=1 \
--block_size=64 \
--batch_size=64 \
--n_layer=8 \
--n_head=8 \
--n_embd=256 \
--max_iters=2000 \
--lr_decay_iters=2000 \
--dropout=0.0 \
--wandb_log=True \
--wandb_group_name='muon' \
--wandb_run_name='muon-lr0.01' \
--lr_finder=False \
--learning_rate=0.01 \


# train loss 1.1326, val loss 1.5803