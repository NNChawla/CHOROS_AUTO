source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CHOROS
# export CHOROS_DATA_ROOT=./srv/CHOROS/data

# Random Search
python training/hparam_search.py --gpu 0 --n_trials 200 --study_name 5_21_26_random_search --storage "postgresql://optuna:choroshps@app.arcadea.us/optuna_choros" \
    --worker_offset 3090 --skip_final --objective posejepa --compile --trial_timeout 7200 --batch_size 256 --compile \
    --trial_epochs 100 --eval_interval 10 --kinematics P --score_mode stable_tail --auc_weight 0.5 --mcc_weight 0.5 --weakest_target_weight 0.25 \
    --score_tail_k 5 --volatility_penalty 0.5 --trend_weight 0.1 --pruner_startup 40 --sampler_startup 9999999 --n_ei_candidates 64