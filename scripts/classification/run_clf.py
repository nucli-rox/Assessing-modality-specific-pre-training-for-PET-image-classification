import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))
import train_clf_WANDB as trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup-config", default="../../configs/classification/setup_clf.yaml"
    )
    parser.add_argument("--run-name", default="run 1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--cv-fold", type=int, default=None)
    parser.add_argument("--minimize_data", type=int, default=None)
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Classification with MAE pre-trained encoder",
    )
    runs = None
    args = parser.parse_args()
    effective_runs = runs or [args.setup_config]
    for setup_cfg in effective_runs:
        run_args = argparse.Namespace(**vars(args))
        run_args.setup_config = setup_cfg

        trainer.main(run_args)


if __name__ == "__main__":
    main()
