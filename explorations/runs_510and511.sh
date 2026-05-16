# 3060_1 5_11_26_search
CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 300 --embed_eval_interval 4 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 10 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 4 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 1024 --latent_loss smooth_l1 --embed_pool mean --lr 3e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 512 --n_heads 8 --n_layers 8 \
--ffn_dim 2048 --max_len 64 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 3060_2 5_11_26_search
CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 300 --embed_eval_interval 4 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 15 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 4 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 1024 --latent_loss smooth_l1 --embed_pool mean --lr 2.5e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 512 --n_heads 8 --n_layers 8 \
--ffn_dim 2048 --max_len 64 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 3090_1 5_11_26_search
CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 300 --embed_eval_interval 4 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 7 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 1024000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 4 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 1024 --latent_loss smooth_l1 --embed_pool mean --lr 3e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 512 --n_heads 8 --n_layers 8 \
--ffn_dim 2048 --max_len 64 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 3090_2 5_11_26_search
CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 300 --embed_eval_interval 4 --batch_size 256 --num_workers 4 --precision bf16 --warmup_epochs 10 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 4 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 1024 --latent_loss smooth_l1 --embed_pool mean --lr 3e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 512 --n_heads 8 --n_layers 8 \
--ffn_dim 2048 --max_len 64 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 4090_1 5_11_26_search
CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir ./srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir outputs/checkpoints \
--epochs 120 --embed_eval_interval 4 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 4 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 4 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 1536 --latent_loss smooth_l1 --embed_pool mean --lr 2.5e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 768 --n_heads 12 --n_layers 8 \
--ffn_dim 3072 --max_len 64 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 4090_2 5_11_26_search
CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir ./srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir outputs/checkpoints \
--epochs 120 --embed_eval_interval 4 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 4 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 4 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 1536 --latent_loss smooth_l1 --embed_pool mean --lr 2e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 768 --n_heads 12 --n_layers 8 \
--ffn_dim 3072 --max_len 64 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 4090_3 5_11_26_search
CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir ./srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir outputs/checkpoints \
--epochs 120 --embed_eval_interval 4 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 4 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 4 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 1536 --latent_loss smooth_l1 --embed_pool mean --lr 1.5e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 768 --n_heads 12 --n_layers 8 \
--ffn_dim 3072 --max_len 64 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 4090_4 5_11_26_search
CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir ./srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir outputs/checkpoints \
--epochs 120 --embed_eval_interval 4 --batch_size 512 --num_workers 4 --precision bf16 --warmup_epochs 4 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 4 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 1536 --latent_loss smooth_l1 --embed_pool mean --lr 1e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 768 --n_heads 12 --n_layers 8 \
--ffn_dim 3072 --max_len 64 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 3060_1 5_10_26_masking_ablations_new_linear
CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 40 --embed_eval_interval 4 --batch_size 512 --num_workers 3 --precision bf16 --warmup_epochs 2 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 384 --latent_loss smooth_l1 --embed_pool mean --lr 6e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 192 --n_heads 4 --n_layers 4 \
--ffn_dim 768 --max_len 128 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 3060_2 5_10_26_masking_ablations_new_linear
CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 40 --embed_eval_interval 4 --batch_size 512 --num_workers 3 --precision bf16 --warmup_epochs 2 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 4 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 384 --latent_loss smooth_l1 --embed_pool mean --lr 6e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 192 --n_heads 4 --n_layers 4 \
--ffn_dim 768 --max_len 128 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 3060_3 5_10_26_masking_ablations_new_linear
CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 40 --embed_eval_interval 4 --batch_size 512 --num_workers 3 --precision bf16 --warmup_epochs 2 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 4 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 384 --latent_loss smooth_l1 --embed_pool mean --lr 6e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 192 --n_heads 4 --n_layers 4 \
--ffn_dim 768 --max_len 64 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 3060_4 5_10_26_masking_ablations_new_linear
CUDA_VISIBLE_DEVICES=1 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 40 --embed_eval_interval 4 --batch_size 512 --num_workers 3 --precision bf16 --warmup_epochs 2 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 384 --latent_loss smooth_l1 --embed_pool mean --lr 6e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 192 --n_heads 4 --n_layers 4 \
--ffn_dim 768 --max_len 256 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 3090_1 5_10_26_masking_ablations_new_linear
CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 80 --embed_eval_interval 4 --batch_size 512 --num_workers 3 --precision bf16 --warmup_epochs 2 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 768 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 384 --n_heads 6 --n_layers 6 \
--ffn_dim 1536 --max_len 128 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 3090_2 5_10_26_masking_ablations_new_linear
CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 80 --embed_eval_interval 4 --batch_size 512 --num_workers 3 --precision bf16 --warmup_epochs 2 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 4 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 768 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 384 --n_heads 6 --n_layers 6 \
--ffn_dim 1536 --max_len 128 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 3090_3 5_10_26_masking_ablations_new_linear
CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 80 --embed_eval_interval 4 --batch_size 512 --num_workers 3 --precision bf16 --warmup_epochs 2 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 4 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 768 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 384 --n_heads 6 --n_layers 6 \
--ffn_dim 1536 --max_len 64 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 3090_4 5_10_26_masking_ablations_new_linear
CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 80 --embed_eval_interval 4 --batch_size 512 --num_workers 3 --precision bf16 --warmup_epochs 2 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 768 --latent_loss smooth_l1 --embed_pool mean --lr 4e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 384 --n_heads 6 --n_layers 6 \
--ffn_dim 1536 --max_len 256 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 4090_1 5_10_26_masking_ablations_new_linear
CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 120 --embed_eval_interval 4 --batch_size 512 --num_workers 3 --precision bf16 --warmup_epochs 2 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 1024 --latent_loss smooth_l1 --embed_pool mean --lr 3e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 512 --n_heads 8 --n_layers 8 \
--ffn_dim 2048 --max_len 128 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 4090_2 5_10_26_masking_ablations_new_linear
CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 120 --embed_eval_interval 4 --batch_size 512 --num_workers 3 --precision bf16 --warmup_epochs 2 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 4 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 1024 --latent_loss smooth_l1 --embed_pool mean --lr 3e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 512 --n_heads 8 --n_layers 8 \
--ffn_dim 2048 --max_len 128 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 4090_3 5_10_26_masking_ablations_new_linear
CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 120 --embed_eval_interval 4 --batch_size 512 --num_workers 3 --precision bf16 --warmup_epochs 2 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 4 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 1024 --latent_loss smooth_l1 --embed_pool mean --lr 3e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 512 --n_heads 8 --n_layers 8 \
--ffn_dim 2048 --max_len 64 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3

# 4090_4 5_10_26_masking_ablations_new_linear
CUDA_VISIBLE_DEVICES=0 python training/train_vr_encoder_pose_jepa.py --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ --out_dir /srv/CHOROS_AUTO/outputs/checkpoints \
--epochs 120 --embed_eval_interval 4 --batch_size 512 --num_workers 3 --precision bf16 --warmup_epochs 2 --val_fraction 0.15 --min_lr 1e-6 --kinematics PVAJ --samples_per_epoch 512000 \
--seed 42 --compile --eval_window_pool stat9 --eval_session_pool mean --eval_split_mode val --patch_size 8 --target_ratio 0.75 \
--target_mode masked_span --n_target_blocks 2 --future_min_gap 2 --future_horizon_min 2 --future_horizon_max 8 --ema_start 0.999 --pred_layers 2 \
--pred_ffn_dim 1024 --latent_loss smooth_l1 --embed_pool mean --lr 3e-4 --sampling_alpha 0.5 --dropout 0.05 --embed_dim 512 --n_heads 8 --n_layers 8 \
--ffn_dim 2048 --max_len 256 --stride_factor 2 --eval_metrics portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3