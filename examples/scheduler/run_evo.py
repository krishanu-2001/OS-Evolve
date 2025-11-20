#!/usr/bin/env python3
"""Evolution entry point for ghOSt CFS scheduler experiments."""

from __future__ import annotations

import argparse
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

TASK_SYS_MSG = """You are a veteran Linux scheduler engineer focused on the
ghOSt userspace implementation of the Completely Fair Scheduler (CFS). Improve
`schedulers/cfs/cfs_scheduler.cc` (and companion headers if needed) to reduce
latency, prevent starvation, and make load balancing + migration decisions more
robust.

Constraints and guidance:
- Preserve the ghOSt agent interfaces and keep channel/Barrier semantics intact.
- Prioritize deterministic behavior inside EVOLVE-BLOCK regions such as
  `SelectTaskRq` and `Migrate`, but you may introduce helper utilities.
- When adding or editing helper files use the `// FILE: relative/path` marker so
  the evaluator can materialize them inside `ghost-userspace`.
- Keep code style consistent with upstream ghOSt (Abseil logging, absl types,
  MutexLock patterns, etc.).
- Document non-obvious heuristics with concise comments.

Evaluation compiles Bazel targets `//:agent_cfs` and `//:cfs_test`. Successful
builds increase the combined_score while failures drop it to zero.

The API definitions are in `schedulers/cfs/cfs_scheduler.h`.

You will be given a set of performance metrics for the program.
Your goal is to maximize the `combined_score` of the program.
Try diverse approaches to solve the problem. Think outside the box.
"""


def build_configs(num_generations: int, results_dir: str):
    job_config = LocalJobConfig(
        eval_program_path="evaluate.py",
    )

    db_config = DatabaseConfig(
        db_path="evolution_db.sqlite",
        num_islands=2,
        archive_size=50,
        elite_selection_ratio=0.3,
        num_archive_inspirations=2,
        num_top_k_inspirations=2,
        migration_interval=8,
        migration_rate=0.15,
        island_elitism=True,
        enforce_island_separation=True,
        parent_selection_strategy="weighted",
        parent_selection_lambda=8.0,
    )

    evo_config = EvolutionConfig(
        task_sys_msg=TASK_SYS_MSG,
        patch_types=["diff", "full", "cross"],
        patch_type_probs=[0.6, 0.3, 0.1],
        num_generations=num_generations,
        max_parallel_jobs=1,
        max_patch_resamples=3,
        max_patch_attempts=3,
        job_type="local",
        language="cpp",
        llm_models=[
            # "gemini-2.5-pro",
            # "gemini-2.5-flash",
            # "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
            # "o4-mini",
            "gpt-5",
            "gpt-5-mini",
        ],
        llm_kwargs=dict(
            temperatures=[0.0, 0.4, 0.8],
            max_tokens=32768,
        ),
        meta_rec_interval=10,
        meta_llm_models=["gpt-5-mini"],
        meta_llm_kwargs=dict(
            temperatures=[0.0],
            max_tokens=8192,
        ),
        init_program_path="initial.cc",
        results_dir=results_dir,
        max_novelty_attempts=3,
        use_text_feedback=False,
    )
    return evo_config, job_config, db_config


def main(num_generations: int, results_dir: str):
    evo_config, job_config, db_config = build_configs(num_generations, results_dir)
    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Shinka evolution to optimize the ghOSt CFS scheduler."
    )
    parser.add_argument("--num_generations", type=int, default=60)
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_cfs_scheduler",
        help="Directory to store evolution artifacts.",
    )
    args = parser.parse_args()
    main(args.num_generations, args.results_dir)
