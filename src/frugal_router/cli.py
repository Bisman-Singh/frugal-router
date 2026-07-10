"""Command-line interface: solve, run, eval, train-predictor, sweep."""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="frugal",
        description="Local-first cascade agent: free local answers, minimal remote tokens.",
    )
    parser.add_argument(
        "--config", default=os.environ.get("FRUGAL_CONFIG", "configs/default.yaml")
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("harness", help="judging-harness mode: tasks.json in, results.json out")
    p.add_argument("--input", default="/input/tasks.json")
    p.add_argument("--output", default="/output/results.json")
    p.add_argument("--time-budget", type=float, default=None)

    p = sub.add_parser("simple", help="minimal passthrough: raw prompt -> strong model -> raw answer")
    p.add_argument("--input", default="/input/tasks.json")
    p.add_argument("--output", default="/output/results.json")

    p = sub.add_parser("solve", help="solve a single task from the command line")
    p.add_argument("--input", required=True)
    p.add_argument("--type", dest="task_type")

    p = sub.add_parser("run", help="solve a JSONL task file, write an answers JSONL")
    p.add_argument("--tasks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ledger", default="runs/ledger.jsonl")

    p = sub.add_parser("eval", help="run a labeled dataset, grade, and log records")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", default="runs/latest")
    p.add_argument(
        "--collect-remote",
        action="store_true",
        help="also query remote for every task so sweeps can replay both branches "
        "(bills tokens during development, never in scoring runs)",
    )

    p = sub.add_parser("train-predictor", help="train the failure predictor from eval records")
    p.add_argument("--records", required=True)
    p.add_argument("--out", default="")

    p = sub.add_parser("sweep", help="replay thresholds over eval records, zero tokens")
    p.add_argument("--records", required=True)
    p.add_argument("--target-acc", type=float, required=True)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--out", default="runs/sweep.csv")

    args = parser.parse_args(argv)
    return _dispatch(args)


def _dispatch(args) -> int:
    if args.command == "simple":
        from .simple import run_simple

        return run_simple(args.input, args.output)

    # Every other subcommand needs the config-driven agent, which the slim
    # scored image does not ship. Fail loudly rather than writing blank
    # answers with a success exit code.
    if not os.path.exists(args.config):
        print(
            f"error: config not found at {args.config!r}. The scored image only "
            f"supports 'frugal simple'; run other commands from the repository.",
            file=sys.stderr,
        )
        return 2

    if args.command == "harness":
        from .harness import run_batch

        return run_batch(
            args.input,
            args.output,
            config_path=args.config,
            time_budget_s=args.time_budget,
        )

    from .config import build_agent, load_settings

    settings = load_settings(args.config)

    if args.command == "solve":
        from .tasks import Task

        agent = build_agent(settings)
        result = agent.solve(Task(id="cli", input=args.input, type=args.task_type))
        print(result.answer)
        print(
            f"[source={result.source} type={result.task_type} "
            f"remote_tokens={result.remote_prompt_tokens + result.remote_completion_tokens} "
            f"path={' > '.join(result.decision_path)}]",
            file=sys.stderr,
        )
        return 0

    if args.command == "run":
        from .ledger import Ledger
        from .tasks import load_tasks

        ledger = Ledger(args.ledger)
        agent = build_agent(settings, ledger=ledger)
        tasks = load_tasks(args.tasks)
        with open(args.output, "w", encoding="utf-8") as fh:
            for task in tasks:
                result = agent.solve(task)
                fh.write(
                    json.dumps({"id": task.id, "answer": result.answer}, ensure_ascii=False)
                    + "\n"
                )
        print(json.dumps(ledger.summary(), indent=2), file=sys.stderr)
        return 0

    if args.command == "eval":
        from .evaluation import run_eval
        from .tasks import load_tasks

        agent = build_agent(settings)
        report = run_eval(
            agent,
            load_tasks(args.dataset),
            out_dir=args.out,
            collect_remote=args.collect_remote,
        )
        print(json.dumps(report["summary"], indent=2))
        return 0

    if args.command == "train-predictor":
        from .evaluation import load_records
        from .predictor import FailurePredictor

        records = [r for r in load_records(args.records) if r.get("local_correct") is not None]
        predictor = FailurePredictor.train(
            [r["input"] for r in records],
            [bool(r["local_correct"]) for r in records],
        )
        out = args.out or settings.predictor_path
        predictor.save(out)
        print(f"trained on {len(records)} examples, saved to {out}")
        return 0

    if args.command == "sweep":
        from .evaluation import load_records
        from .sweep import pareto, sweep, write_csv

        records = load_records(args.records)
        thresholds = [round(0.05 * i, 2) for i in range(0, 20)]
        cutoffs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        rows, recommendation = sweep(
            records, thresholds, cutoffs, args.target_acc, args.margin
        )
        write_csv(rows, args.out)
        print("pareto frontier (tokens ascending):")
        for row in pareto(rows):
            print(f"  {row}")
        print(f"recommended operating point: {recommendation}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
