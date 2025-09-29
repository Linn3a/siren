# Selective Entropy Regularization (SIREN) 🧜‍♀️

This repository provides the official implementation of Selective Entropy Regularization (***SIREN***), introduced in our paper.
***SIREN*** is designed to address the challenges of entropy collapse in RLVR with entropy regularization in context of large reasoning models.
The implementation is built upon the [verl](https://github.com/volcengine/verl) framework, with several key modifications in entropy calculation, aggregation, and objective design.

## Installation ⚙️

We recommend creating a clean conda environment to avoid dependency conflicts.

```
conda create -n siren python=3.10
conda activate siren
pip install -r requirements.txt

# install verl
cd verl
pip install -e .
```


## Usage 🍽️

### Prepare data

```bash
huggingface-cli download --repo-type dataset --resume-download Elliott/Openr1-Math-46k-8192 --local-dir data
```

### Running

We provide example scripts for both training and evaluation.

```bash
# training
bash exp_scripts/siren.sh

# evaluation
bash exp_scripts/eval.sh

```

- The training script (siren.sh) contains default hyperparameters and can be customized according to your experimental setup. 

