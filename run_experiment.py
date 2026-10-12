from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "ablation_suite.json"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "RGBA"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs"
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"


def discover_datasets(root: Path) -> list[str]:
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HoloNet experiments from a JSON suite config")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--datasets", type=str, nargs="*", default=None)
    parser.add_argument("--experiments", type=str, nargs="*", default=None)
    parser.add_argument("--ablation-only", action="store_true")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path, base_dir: Path = PROJECT_ROOT) -> Path:
    return path if path.is_absolute() else (base_dir / path).resolve()


def serialize_value(value: Any) -> str:
    return str(value)


def build_train_command(
    python_executable: str,
    dataset_name: str,
    data_root: Path,
    output_dir: Path,
    train_args: dict[str, Any],
    device: str | None,
) -> list[str]:
    command = [
        python_executable,
        str(TRAIN_SCRIPT),
        "--dataset",
        dataset_name,
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
    ]

    if device:
        command.extend(["--device", device])

    for key, value in train_args.items():
        option_name = key.replace("_", "-")
        option = f"--{option_name}"
        if isinstance(value, bool):
            command.append(option if value else f"--disable-{option_name}")
        elif isinstance(value, list):
            command.append(option)
            command.extend(str(item) for item in value)
        else:
            command.extend([option, serialize_value(value)])

    return command


def load_suite_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_datasets(args: argparse.Namespace, config: dict[str, Any], data_root: Path) -> list[str]:
    if args.datasets:
        return list(args.datasets)
    if args.all_datasets or not config.get("datasets"):
        return discover_datasets(data_root)
    return list(config["datasets"])


def resolve_experiments(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    experiments = list(config.get("experiments", []))

    if args.ablation_only:
        experiments = [experiment for experiment in experiments if experiment["name"].startswith("ablation_")]

    if args.experiments:
        requested = set(args.experiments)
        experiments = [experiment for experiment in experiments if experiment["name"] in requested]
        missing = requested.difference(experiment["name"] for experiment in experiments)
        if missing:
            raise SystemExit(f"unknown experiment names: {', '.join(sorted(missing))}")

    if not experiments:
        raise SystemExit("no experiments selected")

    return experiments


def stream_process_output(command: list[str], log_path: Path, config_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        log_file.write("COMMAND: " + " ".join(command) + "\n")
        log_file.write(f"PROJECT_ROOT: {PROJECT_ROOT}\n")
        log_file.write(f"CONFIG: {config_path}\n")
        log_file.write("LIVE_STDOUT: true\n\n")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        process.stdout.close()
        return process.wait()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir)

    config = load_suite_config(config_path)
    base_train_args = dict(config.get("base_train_args", {}))

    if args.epochs is not None:
        base_train_args["epochs"] = args.epochs
    if args.batch_size is not None:
        base_train_args["batch_size"] = args.batch_size
    if args.num_workers is not None:
        base_train_args["num_workers"] = args.num_workers

    datasets = resolve_datasets(args, config, data_root)
    experiments = resolve_experiments(args, config)

    launcher_root = output_dir / "launcher"
    launcher_root.mkdir(parents=True, exist_ok=True)
    launcher_summary_path = launcher_root / "summary.json"

    summary: list[dict[str, Any]] = []
    for experiment in experiments:
        experiment_name = experiment["name"]
        experiment_train_args = dict(base_train_args)
        experiment_train_args.update(experiment.get("train_args", {}))

        for dataset_name in datasets:
            command = build_train_command(
                python_executable=args.python,
                dataset_name=dataset_name,
                data_root=data_root,
                output_dir=output_dir,
                train_args=experiment_train_args,
                device=args.device,
            )
            log_path = launcher_root / f"{experiment_name}_{dataset_name.lower()}.log"

            run_record = {
                "experiment": experiment_name,
                "dataset": dataset_name,
                "command": command,
                "log_path": str(log_path),
                "status": "pending",
            }

            print(f"[launcher] experiment={experiment_name} dataset={dataset_name}")
            print("[launcher] command:", " ".join(command))

            if args.dry_run:
                run_record["status"] = "dry_run"
                summary.append(run_record)
                continue

            start_time = time.time()
            return_code = stream_process_output(command, log_path, config_path)

            run_record["seconds"] = time.time() - start_time
            run_record["return_code"] = return_code
            run_record["status"] = "completed" if return_code == 0 else "failed"
            summary.append(run_record)
            launcher_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            if return_code != 0 and args.stop_on_error:
                raise SystemExit(f"experiment failed: {experiment_name} on {dataset_name}")

    launcher_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()