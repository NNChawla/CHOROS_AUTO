source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CHOROS

# # 150 epochs
# CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
# --epochs 150 --embed_eval_interval 10 --batch_size 512 --num_workers 6 --precision bf16 --warmup_epochs 12 --val_fraction 0.1 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
# --seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.5 \
# --target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
# --pred_ffn_dim 512 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.0 --embed_dim 256 --n_heads 8 --n_layers 6 \
# --ffn_dim 1024 --max_len 128 --eval_use_best_ckpt
# 
# # target ratio 0.25
# CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
# --epochs 150 --embed_eval_interval 10 --batch_size 512 --num_workers 6 --precision bf16 --warmup_epochs 12 --val_fraction 0.1 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
# --seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.25 \
# --target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
# --pred_ffn_dim 512 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.0 --embed_dim 256 --n_heads 8 --n_layers 6 \
# --ffn_dim 1024 --max_len 128 --eval_use_best_ckpt
# 
# # dropout 0.05
# CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
# --epochs 150 --embed_eval_interval 10 --batch_size 512 --num_workers 6 --precision bf16 --warmup_epochs 12 --val_fraction 0.1 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
# --seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.5 \
# --target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
# --pred_ffn_dim 512 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 256 --n_heads 8 --n_layers 6 \
# --ffn_dim 1024 --max_len 128 --eval_use_best_ckpt

# # 64000 samples per epoch
# CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
# --epochs 150 --embed_eval_interval 10 --batch_size 512 --num_workers 6 --precision bf16 --warmup_epochs 12 --val_fraction 0.1 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 64000 \
# --seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.5 \
# --target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
# --pred_ffn_dim 512 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.0 --embed_dim 256 --n_heads 8 --n_layers 6 \
# --ffn_dim 1024 --max_len 128 --eval_use_best_ckpt