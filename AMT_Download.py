from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="stanford-crfm/music-small-800k",
    local_dir="model/music-small-800k",
    local_dir_use_symlinks=False
)
