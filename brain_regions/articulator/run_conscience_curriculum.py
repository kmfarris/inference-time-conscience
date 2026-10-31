"""CLI for teacher-augmented SNN conscience curriculum generation."""

from __future__ import annotations

import argparse
import json

from brain_regions.articulator.conscience_curriculum import (
    build_conscience_trajectories,
    generate_conscience_curriculum,
    run_conscience_pipeline,
)
from brain_regions.evaluation.local_teacher import LocalOllamaTeacher


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SNN conscience training data")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Run teacher + trajectory pipeline")
    generate.add_argument(
        "--output",
        default="artifacts/conscience_curriculum_v1",
    )
    generate.add_argument("--max-lessons", type=int, default=None)
    generate.add_argument("--no-dilemmas", action="store_true")
    generate.add_argument("--device", default="cpu")

    curriculum = sub.add_parser("curriculum", help="Teacher curriculum only")
    curriculum.add_argument(
        "--output",
        default="artifacts/conscience_curriculum_v1/curriculum",
    )
    curriculum.add_argument("--max-lessons", type=int, default=None)
    curriculum.add_argument("--no-dilemmas", action="store_true")

    trajectories = sub.add_parser("trajectories", help="Build trajectories from curriculum")
    trajectories.add_argument(
        "--curriculum",
        default="artifacts/conscience_curriculum_v1/curriculum",
    )
    trajectories.add_argument(
        "--output",
        default="artifacts/conscience_curriculum_v1/trajectories",
    )
    trajectories.add_argument("--device", default="cpu")

    options = parser.parse_args(argv)
    if options.command == "generate":
        report = run_conscience_pipeline(
            options.output,
            max_lessons=options.max_lessons,
            include_dilemmas=not options.no_dilemmas,
            device=options.device,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if options.command == "curriculum":
        teacher = LocalOllamaTeacher(
            cache_dir=f"{options.output}/../teacher_cache"
        )
        from brain_regions.articulator.conscience_seeds import all_conscience_seeds

        manifest = generate_conscience_curriculum(
            teacher,
            options.output,
            seeds=all_conscience_seeds(include_dilemmas=not options.no_dilemmas),
            max_lessons=options.max_lessons,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if options.command == "trajectories":
        manifest = build_conscience_trajectories(
            options.curriculum,
            options.output,
            device=options.device,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
