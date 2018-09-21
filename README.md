# Intro
Previous mice experiments found spaceflight may harm ovarian health. More analysis is needed to support prolonged spaceflight. We aim to develop a deep learning algorithm and software to efficiently analyze reproductive-cycle data, and release this tool to the scientific community. **This is a work in progress.**

# Install Dependencies (Anaconda)
We use Anaconda to manage our Python environment. The Python dependencies are listed in `environment.yml`. 

To create the conda environment, run:
```bash 
conda env create -f environment.yml
```

# Train a Model
The code for a ResNet transfer-learning model currently exists in the repository. You can train it with the following:

```bash
python train.py -e {experiment_name} -d data/[dataset_name] -m {model} {model_params} -o {optimizer} {optimizer_params} -n {num_epochs} -b {batch_size}
```

Use the `-h` flag for help.

By default, training outputs to `experiments/[experiment_name]/[dataset_name]/[model]/`.

# Contributing
To contribute, either talk to the team in person or shoot us an email (Andrew's is andrewjong87@gmail.com). To propose new changes, create a new branch and submit a PR.
