import argparse
import subprocess
import sys
from pathlib import Path

from experiment_config import DEIT_BACKBONE_NAMES


TRAINERS = {
    "stanford": "train_deit.py",
    "oxfordpets": "train_deit_oxfordpets.py",
}


def main(dataset="both", backbone="all"):
    datasets = tuple(TRAINERS) if dataset == "both" else (dataset,)
    backbones = DEIT_BACKBONE_NAMES if backbone == "all" else (backbone,)
    planned_runs = [(dataset_name, backbone_name) for dataset_name in datasets for backbone_name in backbones]

    print("Planned DeiT experiments:")
    for dataset_name, backbone_name in planned_runs:
        print(f"  dataset={dataset_name}, backbone={backbone_name}")

    failures = []
    for dataset_name, backbone_name in planned_runs:
        script_path = Path(__file__).with_name(TRAINERS[dataset_name])
        command = [sys.executable, str(script_path), "--backbone", backbone_name]
        print(
            f"\nStarting isolated process: dataset={dataset_name}, "
            f"backbone={backbone_name}"
        )
        completed = subprocess.run(command, cwd=script_path.parent, check=False)
        if completed.returncode != 0:
            failures.append((dataset_name, backbone_name, completed.returncode))
            print(
                f"DeiT experiment failed: dataset={dataset_name}, "
                f"backbone={backbone_name}, exit_code={completed.returncode}"
            )

    if failures:
        failed_names = ", ".join(
            f"{dataset_name}/{backbone_name}"
            for dataset_name, backbone_name, _ in failures
        )
        raise RuntimeError(f"Failed DeiT experiments: {failed_names}")


def cli():
    parser = argparse.ArgumentParser(
        description="Run the configured DeiT experiment matrix using protocol v3."
    )
    parser.add_argument(
        "--dataset",
        choices=("stanford", "oxfordpets", "both"),
        default="both",
        help="Dataset scope. The default runs both official datasets.",
    )
    parser.add_argument(
        "--backbone",
        choices=("all",) + DEIT_BACKBONE_NAMES,
        default="all",
        help="Backbone scope. The default runs all six DeiT variants.",
    )
    args = parser.parse_args()
    main(dataset=args.dataset, backbone=args.backbone)


if __name__ == "__main__":
    cli()
