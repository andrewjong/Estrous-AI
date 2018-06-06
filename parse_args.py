
import argparse

MODEL_CHOICES = {
    "cnn_basic": "TODO: CNN BASIC CLASS",
    "resnet_transfer": "TODO: RESNET TRANSFER CLASS",
    "svc_transfer": "TODO: SVC CLASS",
}


def parse_args():
    """ Get args """
    # TODO: Convert these arguments into a configuration file later
    parser = argparse.ArgumentParser(
        description="Run estrous cycle phase-classifer to either train, test, or inference.")

    # TODO maybe the model argument isn't necessary if we can infer the model from load
    parser.add_argument("-m", "--model",
                        required=True,
                        choices=list(MODEL_CHOICES),
                        help=f'Choose which model to use.')

    parser.add_argument("-t", "--task",
                        required=True,
                        choices=["train", "test", "inference"],
                        help='Either "train", "test", or "inference". Train trains on the dataset and outputs train and \
                            validation accuracies. Test evaluates the model on the test portion of the dataset; use this \
                            only when ready for final evaluation! Infer lets the model perform inferences on unseen data.')

    parser.add_argument("-e", "--epochs",
                        type=int,
                        metavar="N",
                        default=50,
                        help='Number of epochs to train for (default: 50). ONLY to be used for "train" task.')

    save_arg = parser.add_argument("-s", "--save",
                                   default="models",
                                   help='Directory to save model parameters (default: "models"). \
                                   ONLY to be used for "train" task. If the directory path does not exist, \
                                   the program will create it.')
    load_arg = parser.add_argument("-l", "--load",
                                   default="models",
                                   help='Directory to load model parameters (default: "models"). \
                            ONLY to be used for "test" or "infer" tasks.')
    inf_arg = parser.add_argument("-i", "--inf",
                                  help="Directory containing images to perform inference on.")

    parser.add_argument("-r", "--results",
                        default="results",
                        help='Directory to store results (default: "results"). \
                            If the directory path does not exist, the program will create it.')
    args = parser.parse_args()

    # Argument checking
    if (args.save != parser.get_default("save")) and args.task != "train":
        raise argparse.ArgumentError(
            save_arg, 'Cannot save a model if "--task" is not "train".')

    if (args.load != parser.get_default("load")) and args.task != "test" and args.task != "inference":
        raise argparse.ArgumentError(
            load_arg, 'Cannot load a model if "--task" is not "test" nor "infer".')

    if (args.inf != parser.get_default("inf")) and args.task != "inference":
        raise argparse.ArgumentError(
            inf_arg, 'Cannot choose a directory to perform inference on if "--task" is not "infer".')

    return args
