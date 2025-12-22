# redteam_runner.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from config import PATHS
from support_bot import SupportBot
from attacker_llm import AttackerLLM, AttackCase, save_attacks_to_file, load_attacks_from_file
from evaluator_llm import EvaluatorLLM, EvalRecord, summarize_results


def load_policies() -> Dict[str, str]:
    """Load all policy files from the policies/ directory."""
    policies_dir = Path(PATHS.policies_dir)
    if not policies_dir.exists():
        raise FileNotFoundError(f"Policies directory not found: {policies_dir}")

    policies: Dict[str, str] = {}
    for p in policies_dir.glob("*.txt"):
        policy_id = p.stem  # e.g. returns, subscription, travel
        policies[policy_id] = p.read_text(encoding="utf-8")
    return policies


def generate_and_save_attacks(num_attacks_per_policy: int = 6) -> None:
    """Generate adversarial prompts for each policy and save to JSON."""
    policies = load_policies()
    attacker = AttackerLLM()
    all_attacks: List[AttackCase] = []
    next_id = 1

    for policy_id, text in policies.items():
        attacks = attacker.generate_attacks_for_policy(
            policy_id=policy_id,
            policy_text=text,
            num_attacks=num_attacks_per_policy,
        )
        for a in attacks:
            a.id = next_id
            next_id += 1
            all_attacks.append(a)

    save_attacks_to_file(all_attacks)


def run_redteam_from_attacks(min_similarity: float | None = None) -> Tuple[Dict, List[EvalRecord]]:
    """
    Run full red-team evaluation using already generated attacks.

    This now supports Streamlit’s dynamic hallucination threshold.
    """
    policies = load_policies()
    bot = SupportBot()
    evaluator = EvaluatorLLM()  # create evaluator

    # apply UI override if provided
    if min_similarity is not None:
        evaluator.min_similarity = min_similarity

    attacks = load_attacks_from_file()
    records: List[EvalRecord] = []

    for attack in attacks:
        policy_text = policies.get(attack.policy_id)
        if policy_text is None:
            continue

        # get model answer
        answer = bot.answer(policy_text=policy_text, user_prompt=attack.prompt)

        # evaluate answer
        rec = evaluator.evaluate(
            test_id=f"case_{attack.id}",
            policy_id=attack.policy_id,
            attack_type=attack.attack_type,
            policy_text=policy_text,
            user_prompt=attack.prompt,
            model_output=answer,
        )
        records.append(rec)

    summary = summarize_results(records)
    return summary, records
