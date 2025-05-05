# This part ensure that we can run the model offine (avoid unstable Internet connections).
# I suggest running on Google Colab (or if you have any better choice, plz call me)
from huggingface_hub import snapshot_download
import os

# The ids are the name of the HuggingFace repos
model2 = "meta-llama/Meta-Llama-3-8B"
tk = "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"  # Created by RY

# Download to the Colab, not in the local computer
save_path = f"/content/models/{model2.replace('/', '_')}"
local_dir = snapshot_download(
    token = tk,
    repo_id=model2,
    repo_type="model",
    local_dir=save_path
)
print(f"Model downloaded to: {local_dir}")
