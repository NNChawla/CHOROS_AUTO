source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CHOROS
# export CHOROS_DATA_ROOT=./srv/CHOROS/data0

python training/hparam_search.py --gpu 1 --n_trials 100 --study_name 5_18_26_search --storage "postgresql://optuna:choroshps@app.arcadea.us/optuna_choros" \
    --worker_offset 3060 --skip_final --objective posejepa --compile