import argparse

kwargs_help = "Pass additional keyword arguments by adding [argument]=[value] "
parser = argparse.ArgumentParser(
    description="Train an estrous cycle cell-phase classifer model."
)

parser.add_argument(
    "-e",
    "--experiment_name",
    help='Name of the experiment. This becomes the subdirectory that the \
    experiment output is stored in, i.e. "experiments/my_experiment/" \
    (default: "unnamed_experiment").',
)
parser.add_argument(
    "-d",
    "--data_dir",
    help='Root directory of dataset to use, with classes separated into \
    separate subdirectories.',
)
parser.add_argument(
    "-m",
    "--model",
    nargs="+",
    help="Choose which model to use, either from torchvision or impelmented \
    here. "
    + kwargs_help
    + "e.g. 'num_classes=4' The list of keyword \
    arguments per model are in the source code. Will add pretrained=True by \
    default.",
)
parser.add_argument(
    "-o",
    "--optimizer",
    nargs="+",
    default=['SGD', "lr=0.001", "momentum=0.9"],
    help="Choose the optimizer from torch.optim to use for training."
    + kwargs_help
    + "e.g. 'lr=0.001'. \
    (default: ['SGD', 'lr=0.001', 'momentum=0.9']).",
)
parser.add_argument(
    "-s",
    "--lr_scheduler",
    nargs="+",
    default=None,
    help="Choose a learning rate scheduler from torch.optim.lr_scheduler. \
    Typically only used with SGD (default: None).",
)
parser.add_argument(
    "-c",
    "--criterion",
    nargs="+",
    default=['CrossEntropyLoss'],
    help="Criterion, i.e. loss function to use. See torch.nn.modules.loss for \
    options.",
)
parser.add_argument(
    "-t",
    "--transfer_technique",
    choices=['finetune', 'fixed', None],
    default='finetune',
    help="Choose between finetune or fixed technique for transfer learning \
    (default: 'finetune').",
)
# TODO: replace the above transfer_technique arg with the code below.
# parser.add_argument(
#     "-f", "--finetune_layers",
#     type=int,
#     default=1,
#     help="Choose how many layers to finetune. Setting 1 makes this a fixed \
#     feature extractor (default: 1)."
# )
parser.add_argument(
    "-n",
    "--num_epochs",
    type=int,
    metavar="N",
    default=50,
    help="Number of epochs to train for (default: 50).",
)
parser.add_argument(
    "-b", "--batch_size", type=int, default=4, help="Select a batch size."
)
parser.add_argument(
    "-l",
    "--load_args",
    nargs="?",
    help="Option to load in train args from a previous train session using \
    that session's \"meta.json\" file. Passed and default args will override \
    loaded args.",
)
parser.add_argument(
    "-p",
    "--use_previous_model",
    action='store_true',
    help="Continue training from the model.pth file created from a previous \
    train run. Must have --load_args argument passed in.",
)
parser.add_argument(
    "--skip_metrics",
    action='store_true',
    help="Choose to NOT run predict.py and metrics.py on the validation-set \
    after training, i.e. do not calculate performance metrics.",
)
parser.add_argument(
    "--overwrite",
    action='store_true',
    help="Skip the prompt that double checks whether to overwrite existing \
    files. Existing files will be overwritten.",
)

train_args = parser.parse_args()

# add pretrained=True to model added args unless the user specified otherwise
if train_args.model:
    if not any("pretrained" in word for word in train_args.model):
        train_args.model.append("pretrained=True")
    # no transfer technique if we're not doing pretraining
    elif "pretrained=false" in (arg.lower() for arg in train_args.model):
        train_args.transfer_technique = None


print(train_args)
