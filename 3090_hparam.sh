source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CHOROS
# export CHOROS_DATA_ROOT=./srv/CHOROS/data0

python training/hparam_search.py --gpu 0 --n_trials 200 --study_name 5_19_26_search --storage "postgresql://optuna:choroshps@app.arcadea.us/optuna_choros" \
    --worker_offset 3090 --skip_final --objective posejepa --compile --trial_timeout 7200 --batch_size 256 \
    --trial_epochs 100 --eval_interval 10 --pruner_startup 5