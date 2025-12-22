# config.py
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ------------------------------
# MODEL SELECTORS
# ------------------------------

def get_support_model() -> str:
    return os.getenv("SUPPORT_BOT_MODEL", "gpt-4.1-nano")

def get_attacker_model() -> str:
    return os.getenv("ATTACKER_MODEL", "gpt-4.1-nano")

def get_eval_model() -> str:
    return os.getenv("EVALUATOR_MODEL", "gpt-4.1-nano")

def get_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ------------------------------
# PATHS
# ------------------------------

class PATHS:
    policies_dir = "policies"
    attacks_file = "attacks/generated_attacks.json"
    output_dir = "output"

# Ensure dirs exist
Path(PATHS.policies_dir).mkdir(exist_ok=True)
Path("attacks").mkdir(exist_ok=True)
Path(PATHS.output_dir).mkdir(exist_ok=True)
