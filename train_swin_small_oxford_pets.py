from train_swin import main as run_swin_experiment


DATASET = "oxfordpets"
BACKBONE = "swin_small"


def main():
    """Train only Swin-Small on Oxford-IIIT Pet using protocol v3."""
    print(f"Starting experiment: dataset={DATASET}, backbone={BACKBONE}")
    run_swin_experiment(dataset_name=DATASET, backbone=BACKBONE)


if __name__ == "__main__":
    main()
