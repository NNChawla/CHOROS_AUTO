source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CHOROS
# export CHOROS_DATA_ROOT=./srv/CHOROS/data

CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 40 --embed_eval_interval 4 --batch_size 512 --num_workers 3 --precision bf16 --warmup_epochs 2 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.25 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 384 --latent_loss smooth_l1 --embed_pool mean --lr 6e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 192 --n_heads 4 --n_layers 4 \
--ffn_dim 768 --max_len 128 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 40 --embed_eval_interval 4 --batch_size 512 --num_workers 3 --precision bf16 --warmup_epochs 2 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 4 --target_ratio 0.25 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 384 --latent_loss smooth_l1 --embed_pool mean --lr 6e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 192 --n_heads 4 --n_layers 4 \
--ffn_dim 768 --max_len 128 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 40 --embed_eval_interval 4 --batch_size 512 --num_workers 3 --precision bf16 --warmup_epochs 2 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 4 --target_ratio 0.25 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 384 --latent_loss smooth_l1 --embed_pool mean --lr 6e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 192 --n_heads 4 --n_layers 4 \
--ffn_dim 768 --max_len 64 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 40 --embed_eval_interval 4 --batch_size 512 --num_workers 3 --precision bf16 --warmup_epochs 2 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.25 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 384 --latent_loss smooth_l1 --embed_pool mean --lr 6e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 192 --n_heads 4 --n_layers 4 \
--ffn_dim 768 --max_len 256 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3