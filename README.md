# zprp_project_25z_group_11

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Data Generators

In order to run generating raw data file for Reber experiment, run:

```bash
py .\zprp_project_25z_group_11\generators\reber.py

```
In order to run generating raw data file for adding or multiplication experiment, run:

```bash
py .\zprp_project_25z_group_11\generators\components.py --task TASK_NAME --num_sequences N --rng RNG
```
#### Arguments:
`--task`: Task to generate data for</br>
* `adding`</br>
* `multiplication` 

`--num_sequences`: Number of sequences to generate (integer)</br>

`--rng`: Random RNG (integer)

## Experiments

### Reber experiment
#### Usage
```bash
py .\zprp_project_25z_group_11\experiments\reber.py --model MODEL_NAME --num-tests NUMBER_OF_TESTS
```
#### Arguments:
`--model`: Model to use
* `LSTM` PyTorch implementation of long short-term memory (LSTM) RNN 
* `GRU`  PyTorch implementation of gated recurrent unit (GRU) RNN
* `LRU`  Gothos/LRU-pytorch implementation of Linear Recurrent Units (LRU)

`--num-tests`: Number of tests to run (default value - 1)

### Adding experiment
#### Usage
```bash
python .\zprp_project_25z_group_11\experiments\adding.py --model MODEL_NAME --data DATA_MODE --rng RNG
```
#### Arguments:
`--model`: Model to use</br>
* `LSTM` PyTorch implementation of long short-term memory (LSTM) RNN </br>
* `GRU`  PyTorch implementation of gated recurrent unit (GRU) RNN </br>
* `LRU`  Gothos/LRU-pytorch implementation of Linear Recurrent Units (LRU)

`--data`: Data source</br>
* `generate` generate new sequences while learning</br>
* `file`     load sequences from file

`--rng`: Random RNG (integer)

### Multiplication experiment
#### Usage
```bash
python .\zprp_project_25z_group_11\experiments\multiplication.py --model MODEL_NAME --data DATA_MODE --rng RNG
```
#### Arguments:
`--model`: Model to use</br>
* `LSTM` PyTorch implementation of long short-term memory (LSTM) RNN </br>
* `GRU`  PyTorch implementation of gated recurrent unit (GRU) RNN </br>
* `LRU`  Gothos/LRU-pytorch implementation of Linear Recurrent Units (LRU)

`--data`: Data source</br>
* `generate` generate new sequences while learning</br>
* `file`     load sequences from file

`--rng`: Random RNG (integer)

## Tests

### Usage

Adding:

```bash
pytest .\tests\components_adding.py
````

Multiplication:

```bash
pytest .\tests\components_multiplication.py 
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Documentation for the project.
│
├── notebooks          <- Jupyter notebooks.
│   └── 1.0-lru-sanity-check.ipynb
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         zprp_project_25z_group_11 and configuration for tools like black
│
├── reports            <- Generated analysis.
│   ├── logs           <- Events saved from experiments
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── tests              <- Tests for generators
│   ├── components_adding_test.py
│   └── components_multiplication_test.py
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│
└── zprp_project_25z_group_11   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes zprp_project_25z_group_11 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── experiments                
    │   ├── __init__.py 
    │   ├── reber.py            <- Code to run reber grammar experiment    
    │   ├── adding.py           <- Code to run adding experiment                     
    │   └── multiplication.py   <- Code to run multiplication experiment 
    │
    └── generators                
        ├── __init__.py 
        ├── reber.py            <- Code to generate reber grammar data       
        └── components.py       <- Code to generate sequences for adding and multiplication experiment
    
```

--------

