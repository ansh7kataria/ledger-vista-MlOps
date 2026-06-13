"""
hosting.py
Pushes the deployment files (app.py, requirements.txt, Dockerfile) to the
Hugging Face Space that serves the Streamlit frontend.

This is the last pipeline step: once the best model is registered, this puts a
working UI in front of it so anyone with the Space link can score a customer.
"""

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

HF_TOKEN = os.getenv("HF_TOKEN")
SPACE_REPO = "iamanshkataria/tourism-package-app"

api = HfApi(token=HF_TOKEN)

# Create the Space if needed. Docker SDK because the project ships a Dockerfile.
try:
    create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="docker",
        token=HF_TOKEN,
        exist_ok=True,
    )
    print(f"Space ready: {SPACE_REPO}")
except HfHubHTTPError as e:
    print(f"Space check/create returned: {e}")

# Push each deployment file up to the Space root.
for fname in ["app.py", "requirements.txt", "Dockerfile"]:
    api.upload_file(
        path_or_fileobj=f"tourism_project/deployment/{fname}",
        path_in_repo=fname,
        repo_id=SPACE_REPO,
        repo_type="space",
    )
print("Deployment files pushed to the Hugging Face Space.")
