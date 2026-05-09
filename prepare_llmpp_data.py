#!/usr/bin/env python3
"""
prepare_llmpp_data.py — Convert LLM+P dataset to LAPIS benchmark format.

Reads from:
    third-party/llm-pddl/domains/{domain}/
        domain.nl       (domain description)
        domain.pddl     (PDDL domain)
        pXX.nl          (problem description)
        pXX.pddl        (ground-truth PDDL problem)

Writes to:
    data/llmpp/{domain}/{problem_id}/
        nl              (domain.nl + pXX.nl concatenated)
        domain.pddl     (symlink or copy)
        problem.pddl    (copy of pXX.pddl)
"""

import shutil
import sys
from pathlib import Path

LLMPP_DOMAINS = ["barman", "blocksworld", "floortile", "grippers", "storage", "termes", "tyreworld"]
N_PROBLEMS = 20

REPO_ROOT = Path(__file__).parent
SRC_ROOT = REPO_ROOT / "third-party" / "llm-pddl" / "domains"
DST_ROOT = REPO_ROOT / "data" / "llmpp"


def prepare_domain(domain: str, overwrite: bool = False):
    src_dir = SRC_ROOT / domain
    if not src_dir.exists():
        print(f"  SKIP: {src_dir} not found")
        return 0

    domain_nl = (src_dir / "domain.nl").read_text()
    domain_pddl = src_dir / "domain.pddl"

    count = 0
    for i in range(1, N_PROBLEMS + 1):
        prob_id = f"p{i:02d}"
        prob_nl_file = src_dir / f"{prob_id}.nl"
        prob_pddl_file = src_dir / f"{prob_id}.pddl"

        if not prob_nl_file.exists() or not prob_pddl_file.exists():
            continue

        dst_dir = DST_ROOT / domain / prob_id
        if dst_dir.exists() and not overwrite:
            count += 1
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)

        # Concatenate domain.nl + problem.nl as the NL input
        prob_nl = prob_nl_file.read_text()
        nl_text = domain_nl.rstrip() + "\n\n" + prob_nl.rstrip() + "\n"
        (dst_dir / "nl").write_text(nl_text)

        # Copy domain.pddl and problem.pddl
        shutil.copy2(domain_pddl, dst_dir / "domain.pddl")
        shutil.copy2(prob_pddl_file, dst_dir / "problem.pddl")

        count += 1

    return count


def main():
    domains = sys.argv[1:] if len(sys.argv) > 1 else LLMPP_DOMAINS
    overwrite = "--overwrite" in domains
    if overwrite:
        domains = [d for d in domains if d != "--overwrite"]

    print(f"Preparing LLM+P data in {DST_ROOT}")
    for domain in domains:
        n = prepare_domain(domain, overwrite=overwrite)
        print(f"  {domain}: {n} problems prepared")


if __name__ == "__main__":
    main()
