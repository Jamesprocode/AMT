from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="stanford-crfm/music-large-800k",
    local_dir="model/music-large-800k",
    local_dir_use_symlinks=False
)
