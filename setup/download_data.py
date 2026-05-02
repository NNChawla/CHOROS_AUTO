from huggingface_hub import snapshot_download

hf_token = "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"

# Replace with your specific repo ID and desired local path
repo_id = "nnchawla/CHOROS"
local_dir = "./data"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",       # Required for dataset repos
    local_dir=local_dir,       # Downloads to this specific folder
    token=hf_token
)