source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CHOROS
# export CHOROS_DATA_ROOT=./srv/CHOROS/data0

python training/hparam_search.py --gpu 0 --n_trials 100 --study_name 5_15_26_search --storage "postgresql://optuna:choroshps@app.arcadea.us/optuna_choros" \
    --worker_offset 0 --skip_final --objective posejepa --compile