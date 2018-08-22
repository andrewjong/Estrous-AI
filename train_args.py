import argparse
from src.model_choices import TRAIN_MODEL_CHOICES

parser = argparse.ArgumentParser(
    description="Train an estrous cycle cell-phase classifer model.")

parser.add_argument(
    "model",
    choices=list(TRAIN_MODEL_CHOICES),
    nargs="?",
    help=f'Choose which model to use.')
parser.add_argument(
    "-e",
    "--experiment_name",
    help='Name of the experiment. This becomes the \
                    subdirectory that the experiment output is stored in, \
                    i.e. "experiments/my_experiment/" (default: \
                    "unnamed_experiment").')
parser.add_argument(
    "-d",
    "--data_dir",
    help='Root directory of dataset to use, with classes \
                    separated into separate subdirectories.')
parser.add_argument(
    "-n",
    "--num_epochs",
    type=int,
    metavar="N",
    default=50,
    help='Number of epochs to train for (default: 50).')
parser.add_argument(
    "-b", "--batch_size", default=4, type=int, help="Select a batch size.")
parser.add_argument(
    "-a",
    "--added_args",
    nargs="+",
    default=[],
    help="Pass addiitional arguments for instantiating the \
                    chosen model.")
parser.add_argument(
    "-l",
    "--load_args",
    nargs="?",
    help="Option to load in train args from a previous \
                    train session using that session's \"meta.json\" \
                    file. Passed and default args will override loaded \
                    args.")
parser.add_argument(
    "-p",
    "--use_previous_model",
    action='store_true',
    help="Continue training from the model.pth file \
                    created from a previous train run. Must have \
                    --load_args argument passed in.")
parser.add_argument(
    "--skip_metrics",
    action='store_true',
    help="Choose to NOT run predict.py and metrics.py \
                        on the validation-set after training, i.e. do not \
                        calculate performance metrics.")
parser.add_argument(
    "--overwrite",
    action='store_true',
    help="Skip the prompt that double checks whether to \
                    overwrite existing files. Existing files will be \
                    overwritten.")

train_args = parser.parse_args()
