# model_building/data_register.py
import os
from huggingface_hub import HfApi, create_repo, upload_file
from huggingface_hub.utils import RepositoryNotFoundError
import sys

# --- CONFIG ---
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN".upper())
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN environment variable is not set. Set it in GitHub Actions secrets as HF_TOKEN.")

api = HfApi(token=HF_TOKEN)
# Use exactly this repo id (you provided it)
repo_id = "AshwiniBokade/Tourism-Project"
repo_type = "dataset"

# --- Ensure repo exists ---
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating it...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=HF_TOKEN)
    print(f"Dataset repo '{repo_id}' created.")

# --- Locate local data folder (try several possible locations) ---
candidates = [
    "Tourism_project/data",
    "tourism_project/data",
    "data",
    "Tourism_project_git/data",
    "tourism_project/data"  # keep common variants
]

found = None
for p in candidates:
    if os.path.isdir(p):
        found = p
        break

# Also check for CSV files in repo root
if not found:
    csvs_in_root = [f for f in os.listdir(".") if f.lower().endswith(".csv")]
    if csvs_in_root:
        found = "root_csvs"

if not found:
    print("ERROR: No data directory found. Tried these paths:", candidates)
    print("Also no CSVs found at repo root.")
    raise SystemExit(1)

# --- Uploading ---
if found == "root_csvs":
    csvs = csvs_in_root
    print("Found CSVs at repo root:", csvs)
    for fname in csvs:
        print("Uploading", fname)
        api.upload_file(
            path_or_fileobj=fname,
            path_in_repo=os.path.basename(fname),
            repo_id=repo_id,
            repo_type=repo_type,
        )
else:
    print("Found data directory at:", found)
    # if folder contains many files, upload them one by one
    for root, dirs, files in os.walk(found):
        rel_root = os.path.relpath(root, found)
        for fn in files:
            local_path = os.path.join(root, fn)
            # path_in_repo preserves folder structure inside dataset repo
            if rel_root == ".":
                path_in_repo = fn
            else:
                path_in_repo = os.path.join(rel_root, fn)
            print(f"Uploading {local_path} -> {path_in_repo}")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
            )

print("Upload complete.")
