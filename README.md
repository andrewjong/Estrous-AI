This repository contains code for classifying estrous cycle phases in cells. We use various machine learning algorithms for this task.

# Work In Progress. More information to come.

# Install Dependencies (Anaconda)
We use Anaconda to manage our Python environment. Download Anaconda here: https://www.anaconda.com/download/

Once Anaconda is installed, create the envrionment from the dependencies file. The Python dependencies are listed   in `environment.yml`. 

From the project root, run: 
``` conda env create -f environment.yml```.

# Getting the Data
Contact Josh or Andrew for the data. Once the data is obtained, use the `data_cleaning/sort_and_split.py` to split the data into train/validation/test sets organized by label.

```python sort_and_split.py labels.csv -s 70 15 15```

# Train a Model
The code for a ResNet18 transfer-learning by finetuning model currently exists in the repository. You can train it with the following:
```python train.py resnet18_transfer```.
