Previous mice experiments found spaceflight may harm ovarian health. More analysis is needed to support prolonged spaceflight. We aim to develop a deep learning algorithm and software to efficiently analyze reproductive-cycle data, and release this tool to the scientific community. **This is a work in progress.**

# Install Dependencies (Anaconda)
We use Anaconda to manage our Python environment. Download Anaconda here: https://www.anaconda.com/download/

Once Anaconda is installed, create the envrionment from the dependencies file. The Python dependencies are listed   in `environment.yml`. 

To create the conda environment, run: 
``` conda env create -f environment.yml```.

# Codebase Structure
TODO

# Getting the Data
Contact Josh or Andrew for the source data. This contains all the images in non-sorted fashion. Once the images are obtained, use the `data_cleaning/sort_and_split.py` to split the data into train/validation/test sets organized by label. This formats the dataset for training. The labels file is `data_cleaning/labels.csv`.

**EASY COPY PASTE COMMAND:**

Create 4-class dataset: 
```python data_cleaning/sort_and_split.py data_cleaning/labels.csv [PATH/TO/LavageJPGS] data/4_class -e 40x art bad```

Create binary "diestrus vs. all" dataset: 
```python data_cleaning/sort_and_split.py data_cleaning/labels.csv [PATH/TO/LavageJPGs] data/die_vs_all -g 123 4 -e 40x art bad```

**Command explanation**
```python data_cleaning/sort_and_split.py data_cleaning/labels.csv [path/to/input/source_dataset] data/[output_dataset_name] -g [group1] [group2] -e [exclude keywords]```

- Group class labels together using the -g flag, e.g. "-g 123 4" groups diestrus (4) vs everything else (123). This allows flexibility for how to group different labels.
- Additionally, exclude keywords using the -e flag. Suggested usage is "-e 40x art", as this will remove the 40x zoomed images and artistically-edited images.

The above command outputs a structure like this:

```bash
├── data
│   ├── [sorted_dataset_name]
│   │   ├── test
│   │   │   ├── diestrus
│   │   │   ├── estrus
│   │   │   ├── metestrus
│   │   │   └── proestrus
│   │   ├── train
│   │   │   ├── diestrus
│   │   │   └── ..
│   │   ├── val
│   │   │   ├── diestrus
└───└───└───└── ..
```
# Viewing Image Transformations
The dataloader performs transformations on-the-fly to perform data augmentation. These can be viewed with the following command:

```python view_transforms.py [path/to/dataset/] -n [number_to_view]```

# Train a Model
The code for a ResNet transfer-learning model currently exists in the repository. You can train it with the following:

```python train.py resnet_transfer -a {18|34|50|101|152} {finetune|fixed} -d data/[sorted_dataset_name]```.

The `-a` flag passes in architectural hyperparameters to the chosen model. The `-d` flag selects the dataset to train on, created by `sort_and_split.py`. Use the `-h` flag for help.

By default, training outputs to `experiments/unnamed_experiment/[sorted_dataset_name]/[model]/`.

# Analyzing Model Performance
We want to see the exact predictions vs. true labels of the model for future F1 score and confusion matrix analysis.

To output model predictions, run ```python predict.py [experiments/experiment_name/dataset/model_name/]```

This loads the model using the `model.pth` and `meta.json` file, then runs the model and outputs to `predictions.csv`.

# Exporting Models
Run the `./export_models` bash script to copy the `*pth` files under `experiments/` to a separate folder called `models/`. 

This is because model files are huge and won't fit on GitHub. As such, `*.pth` is currently in the .gitignore. The export script allows us to store the models elsewhere.

# Contributing
Please check out the Projects tab. There's a lot to do! To contribute, either talk to the team in person or shoot us an email (Andrew's is andrewjong87@gmail.com). To propose new changes, create a new branch and submit a PR.
