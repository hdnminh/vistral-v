from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="/mnt/disks/dev/data/images/wit",
    path_in_repo="images/wit",
    repo_id="Vi-VLM/Vista",
    repo_type="dataset",
    allow_patterns="images_part_*",
)