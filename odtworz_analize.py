"""Reproduce thesis analysis from saved results without loading or training models."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def run(name):
    subprocess.run([sys.executable, str(ROOT / "analiza_wynikow" / name)], cwd=ROOT, check=True)

def main():
    run("generate_v5_assets.py")
    run("generate_v5_performance.py")
    for version in ("v6", "v7"):
        folder = ROOT / "analiza_wynikow" / (version + "_assets")
        folder.mkdir(exist_ok=True)
        for name in ("verified_results.json", "performance_details.json"):
            shutil.copy2(ROOT / "analiza_wynikow/v5_assets" / name, folder / name)
    run("generate_v6_figures.py")
    run("generate_v7_classification_metrics.py")
    run("generate_v7_f1_figure.py")
    run("generate_v7_checkpoint_comparison.py")

if __name__ == "__main__":
    main()
