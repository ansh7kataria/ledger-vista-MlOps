"""
data_register.py
Registers the raw tourism dataset on the Hugging Face dataset space.

Run order in the pipeline: this is step 1. It takes the local raw CSV that
lives in tourism_project/data/ and uploads it to a HF dataset repo so that
every later step pulls from one versioned source of truth instead of a file
sitting on someone's laptop.
"""

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

# ---- config ----
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "iamanshkataria/tourism-package-prediction"
LOCAL_CSV = "tourism_project/data/tourism.csv"

api = HfApi(token=HF_TOKEN)

# Create the dataset repo if it isn't there yet. exist_ok keeps reruns quiet.
try:
    create_repo(repo_id=DATASET_REPO, repo_type="dataset", token=HF_TOKEN, exist_ok=True)
    print(f"Dataset repo ready: {DATASET_REPO}")
except HfHubHTTPError as e:
    print(f"Repo check/create returned: {e}")

# Push the raw file up. path_in_repo keeps it under a clean name on the hub.
api.upload_file(
    path_or_fileobj=LOCAL_CSV,
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset",
)
print("Raw dataset uploaded to the Hugging Face dataset space.")
