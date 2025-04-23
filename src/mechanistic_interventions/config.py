import os
from dotenv import load_dotenv

# Only load a .env file in real runs, not during pytest
if "PYTEST_CURRENT_TEST" not in os.environ:
    load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set — copy .env.example to .env and paste your token")