import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))  # point to repo root
print(sys.path)
import train_clf_WANDB as trainer
import argparse

from nucli_train.models.builders import build_model, MODEL_REGISTRY
from nucli_train.training import Trainer
import mlflow
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]  # repo root above scripts/
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.MIPdataset import build_mip_data
import tasks.models.convnext_MAE_clf
import tasks.models.resnet_clf
import src.nets.convnext
from src.val import evaluator_MAE


# recreate the parser
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup-config", default="../../configs/tasks/setup_MAEclf.yaml"
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--cv-fold", type=int, default=None)
    parser.add_argument("--minimize_data", type=int, default=None)
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="ConvNeXt benchmark",
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
