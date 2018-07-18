This repository contains code for classifying estrous cycle phases in cells. We use various machine learning algorithms for this task. **This is a work in progress.**

# Install Dependencies (Anaconda)
We use Anaconda to manage our Python environment. Download Anaconda here: https://www.anaconda.com/download/

Once Anaconda is installed, create the envrionment from the dependencies file. The Python dependencies are listed   in `environment.yml`. 

From the project root, run: 
``` conda env create -f environment.yml```.

# Getting the Data
Contact Josh or Andrew for the source data. This contains all the images in non-sorted fashion. Once the data is obtained, use the `data_cleaning/sort_and_split.py` to split the data into train/validation/test sets organized by label. This formats the dataset for training.

```python sort_and_split.py labels.csv data/[sorted_dataset_name] -d [path/to/source/dataset]```

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

# Train a Model
The code for a ResNet transfer-learning model currently exists in the repository. You can train it with the following:

```python train.py resnet_transfer -a {18|34|50|101|152} {finetune|fixed} -d data/[sorted_dataset_name]```.

The `-a` flag passes in architectural hyperparameters to the chosen model. The `-d` flag selects the dataset to train on, created by `sort_and_split.py`. Use the `-h` flag for help.

By default, training outputs to `experiments/unnamed_experiment/[sorted_dataset_name]/[model]/`.

# Contributing
Please check out the Projects tab. There's a lot to do! To contribute, either talk to the team in person or shoot us an email (Andrew's is andrewjong87@gmail.com).
