source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CHOROS
# export CHOROS_DATA_ROOT=./srv/CHOROS/data

# 70/30vaj
CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 150 --embed_eval_interval 10 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 12 --val_fraction 0.1 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.25 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 512 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 256 --n_heads 8 --n_layers 6 \
--ffn_dim 1024 --max_len 128 --eval_use_best_ckpt --stride_factor 2 --context_group_mask_schedule "70:" "30:VAJ"

# 70/30paj
CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 150 --embed_eval_interval 10 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 12 --val_fraction 0.1 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.25 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 512 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 256 --n_heads 8 --n_layers 6 \
--ffn_dim 1024 --max_len 128 --eval_use_best_ckpt --stride_factor 2 --context_group_mask_schedule "70:" "30:PAJ"

# 70/30pva
CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 150 --embed_eval_interval 10 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 12 --val_fraction 0.1 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.25 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 512 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 256 --n_heads 8 --n_layers 6 \
--ffn_dim 1024 --max_len 128 --eval_use_best_ckpt --stride_factor 2 --context_group_mask_schedule "70:" "30:PVA"

# # 70/30pvj
# CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
# --epochs 150 --embed_eval_interval 10 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 12 --val_fraction 0.1 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
# --seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.25 \
# --target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
# --pred_ffn_dim 512 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 256 --n_heads 8 --n_layers 6 \
# --ffn_dim 1024 --max_len 128 --eval_use_best_ckpt --stride_factor 2 --context_group_mask_schedule "70:" "30:PVJ"
# 
# # 60/3p/3v/3a/3j/3pv/3pa/3pj/3va/3vj/3aj/3pva/3pvj/3paj/3vaj
# CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
# --epochs 150 --embed_eval_interval 10 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 12 --val_fraction 0.1 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
# --seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.25 \
# --target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
# --pred_ffn_dim 512 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 256 --n_heads 8 --n_layers 6 \
# --ffn_dim 1024 --max_len 128 --eval_use_best_ckpt --stride_factor 2 --context_group_mask_schedule "60:" "3:P" "3:V" "3:A" "3:J" "3:PV" "3:PA" "3:PJ" "3:VA" "3:VJ" "3:AJ" "3:PVA" "3:PVJ" "3:PAJ" "3:VAJ"
# 
# # 65/5v/5a/5j/5va/5vj/5aj/5vaj
# CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
# --epochs 150 --embed_eval_interval 10 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 12 --val_fraction 0.1 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
# --seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.25 \
# --target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
# --pred_ffn_dim 512 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 256 --n_heads 8 --n_layers 6 \
# --ffn_dim 1024 --max_len 128 --eval_use_best_ckpt --stride_factor 2 --context_group_mask_schedule "65:" "5:V" "5:A" "5:J" "5:VA" "5:VJ" "5:AJ" "5:VAJ"