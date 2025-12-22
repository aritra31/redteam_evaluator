# main.py
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import time
from typing import List

from config import PATHS
from redteam_runner import generate_and_save_attacks, run_redteam_from_attacks
from evaluator_llm import EvalRecord, eval_record_to_dict, compute_robustness_score


def write_json_report(summary: dict, records: List[EvalRecord]) -> None:
    """Write JSON report to output/ for offline inspection."""
    out_dir = Path(PATHS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.time().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"redteam_report_{ts}.json"

    robustness = compute_robustness_score(summary)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": ts,
                "robustness_score": robustness,
                "summary": summary,
                "cases": [eval_record_to_dict(r) for r in records],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[OK] JSON report written to: {json_path}")
    print(f"[INFO] Bot Robustness Score: {robustness}/100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SupportBot Red-Team Evaluator")
    parser.add_argument(
        "--mode",
        choices=["generate_attacks", "run_eval"],
        required=True,
        help="generate_attacks: create adversarial prompts; run_eval: run full stress test.",
    )
    parser.add_argument(
        "--num-per-policy",
        type=int,
        default=6,
        help="Number of attacks to generate per policy (for generate_attacks).",
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.70,
        help="Min cosine similarity threshold for hallucination suspicion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "generate_attacks":
        print("[INFO] Generating adversarial prompts...")
        generate_and_save_attacks(num_attacks_per_policy=args.num_per_policy)
        print("[OK] Saved attacks JSON.")
    elif args.mode == "run_eval":
        print("[INFO] Running red-team evaluation...")
        summary, records = run_redteam_from_attacks(min_similarity=args.min_sim)
        write_json_report(summary, records)


if __name__ == "__main__":
    main()
