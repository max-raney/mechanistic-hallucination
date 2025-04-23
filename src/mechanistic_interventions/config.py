# src/config.py
import os
from dotenv import load_dotenv

# look for a .env file in the repo root
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set — copy .env.example to .env and paste your token")
