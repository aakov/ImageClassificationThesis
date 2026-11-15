import argparse
import os
import subprocess
import sys

from train_swin import TRAINING_CONFIGS, main as run_swin_experiment


BACKBONE = "swin_base"


def main(dataset_name="stanford"):
    """Run one Swin-Base experiment through the shared protocol v3 pipeline."""
    run_swin_experiment(dataset_name=dataset_name, backbone=BACKBONE)


def cli():
    parser = argparse.ArgumentParser(
        description="Run the separate Swin-Base experiment using protocol v3."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--dataset",
        choices=tuple(TRAINING_CONFIGS),
        default="stanford",
    )
    selection.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run Stanford Cars and Oxford-IIIT Pet sequentially.",
    )
    args = parser.parse_args()

    datasets = tuple(TRAINING_CONFIGS) if args.all_datasets else (args.dataset,)
    print("Planned separate Swin-Base experiments:")
    for dataset_name in datasets:
        print(f"  dataset={dataset_name}, backbone={BACKBONE}")

    if len(datasets) == 1:
        main(datasets[0])
        return

    failures = []
    for dataset_name in datasets:
        completed = subprocess.run(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--dataset",
                dataset_name,
            ],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=False,
        )
        if completed.returncode != 0:
            failures.append((dataset_name, completed.returncode))
            print(
                f"Swin-Base experiment failed for {dataset_name}: "
                f"exit_code={completed.returncode}"
            )

    if failures:
        names = ", ".join(dataset_name for dataset_name, _ in failures)
        raise RuntimeError(f"Failed Swin-Base experiments: {names}")


if __name__ == "__main__":
    cli()
