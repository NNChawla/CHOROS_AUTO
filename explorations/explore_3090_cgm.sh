source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CHOROS
# export CHOROS_DATA_ROOT=./srv/CHOROS/data

# 65/35vaj
CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 150 --embed_eval_interval 10 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 12 --val_fraction 0.1 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.25 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 512 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 256 --n_heads 8 --n_layers 6 \
--ffn_dim 1024 --max_len 128 --eval_use_best_ckpt --stride_factor 2 --context_group_mask_schedule "65:" "35:VAJ"

# 60/40vaj
CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 150 --embed_eval_interval 10 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 12 --val_fraction 0.1 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.25 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 512 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 256 --n_heads 8 --n_layers 6 \
--ffn_dim 1024 --max_len 128 --eval_use_best_ckpt --stride_factor 2 --context_group_mask_schedule "60:" "40:VAJ"

# 50/50vaj
CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 150 --embed_eval_interval 10 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 12 --val_fraction 0.1 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.25 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 512 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 256 --n_heads 8 --n_layers 6 \
--ffn_dim 1024 --max_len 128 --eval_use_best_ckpt --stride_factor 2 --context_group_mask_schedule "50:" "50:VAJ"

# # 70/10a/10j/10aj
# CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
# --epochs 150 --embed_eval_interval 10 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 12 --val_fraction 0.1 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
# --seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.25 \
# --target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
# --pred_ffn_dim 512 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 256 --n_heads 8 --n_layers 6 \
# --ffn_dim 1024 --max_len 128 --eval_use_best_ckpt --stride_factor 2 --context_group_mask_schedule "70:" "10:A" "10:J" "10:AJ"
# 
# # 70/10p/10v/10pv
# CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
# --epochs 150 --embed_eval_interval 10 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 12 --val_fraction 0.1 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
# --seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.25 \
# --target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
# --pred_ffn_dim 512 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 256 --n_heads 8 --n_layers 6 \
# --ffn_dim 1024 --max_len 128 --eval_use_best_ckpt --stride_factor 2 --context_group_mask_schedule "70:" "10:P" "10:V" "10:PV"