import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix
from src.utils import SORT_BY_PHASE_FN

CONFUSION_MATRIX_NAME = "confusion_matrix.png"
# orders the phases according to the Estrous cycle by first letter
# (*p*roestrus, *e*strus, etc)


def create_all_metrics(predictions_files, outdir, name_prefix=""):
    """Create all possible performance metric and write them to the specified
    output directory.

    Arguments:
        predictions_file(s) {iterable or string} -- files or file to to create
                                                    metrics for
        outdir {string} -- the output directory path

    Keyword Arguments:
        name_prefix {string} -- prefix the file output names if desired
                             (default: empty string)
    """

    if isinstance(predictions_files, str):
        predictions_files = [predictions_files]

    for f in predictions_files:
        cm_outfile = os.path.join(outdir, name_prefix + CONFUSION_MATRIX_NAME)
        create_confusion_matrix_plots(f, cm_outfile)
        # create other metrics


def create_confusion_matrix_plots(predictions_file, out_file):
    """Read a predictions file to output standard and normalized confusion
    matrix files. Normalized file appends "_normalized" to the out file name.

    Arguments:
        predictions_file {string} -- path to predictions csv file, generated by
                                     predict.py
        out_file {string} -- path to output file for image
    """

    df = pd.read_csv(predictions_file)
    try:
        df = df.drop(columns=["not_diestrus"])
    except KeyError:
        pass
    # classes in between the file name (first column) and predicted/label
    # (last two columns)
    classes = list(df.columns[1:-2])
    if len(classes) == 2:
        # sort alphabetical for diestrus vs pro-est-met, or met-die vs pro-est
        classes.sort()
    else:
        # since the first letters for phases are all different, sort using this
        classes.sort(key=SORT_BY_PHASE_FN)

    cm = confusion_matrix(df['label'], df['predicted'], labels=classes)
    # plot standard confusion matrix
    fig = plot_confusion_matrix(cm, classes)
    fig.savefig(out_file)
    # plot normalized confusion matrix
    fig_normalized = plot_confusion_matrix(cm, classes, normalize=True)
    # append "_normalized" to file name
    norm_outfile_name = out_file[:-4] + "_normalized" + out_file[-4:]
    fig_normalized.savefig(norm_outfile_name)


def plot_confusion_matrix(
    confusion_matrix,
    class_names,
    normalize=False,
    figsize=(10, 7),
    fontsize=14,
    cmap='Blues',
):
    """Plots a labeled and colored confusion matrix image. True labels are on
    the y-axis and predicted labels on the x-axis. A colorbar is placed on the
    right.

    Extended from "shaypal5/confusion_matrix_pretty_print.py" on GitHub gists.

    Arguments:
        confusion_matrix {numpy.array} -- array to use for the confusion matrix
        class_names {iterable} -- ordered names to use for classes

    Keyword Arguments:
        normalize {bool} -- whether to normalize (default: {False})
        figsize {tuple} -- how big to make the figure (default: {(10, 7)})
        fontsize {int} -- how big to make the label text (default: {14})
        cmap {string} -- the Seaborn color mapping to use(default: 'Blues')

    Returns:
        matplotlib.figure.Figure -- the resulting confusion matrix figure
    """

    # normalize (along axis 1, i.e. horizontally) if requested
    if normalize:
        confusion_matrix = (
            confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]
        )
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0, None
    # make a dataframe with rows and columns labeled as class names
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)

    fig = plt.figure(figsize=figsize)  # make a figure
    fmt = "0.2f" if normalize else "d"  # number format
    # make the confusion matrix as a heatmap for colors
    heatmap = sns.heatmap(df_cm, vmin=vmin, vmax=vmax, annot=True, fmt=fmt, cmap=cmap)

    # set our axis tick labels for the classes
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize
    )
    # set the x and y axis label
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()  # make everything fit properly
    return fig


# TODO: Other metrics


if __name__ == "__main__":
    # setup parser if running script as standalone
    parser = argparse.ArgumentParser(
        description="Calculate and write performance metrics for model \
        predictions."
    )
    parser.add_argument(
        "predictions_file",
        help='Predictions file to generate metrics for, \
                        typically "*_predictions.csv".',
    )
    args = parser.parse_args()
    # output to the same directory that the file lives in
    outdir = os.path.dirname(args.predictions_file)
    # get whether "val" or "test" based on the filename prefix
    basename = os.path.basename(args.predictions_file)
    prefix = basename.split("_")[0] + "_"
    # output metrics using cmd line args
    print("Creating metrics...")
    print("Writing to directory: " + outdir)
    create_all_metrics(args.predictions_file, outdir, prefix)
    print("Done.")
