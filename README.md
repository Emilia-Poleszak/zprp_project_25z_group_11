# zprp_project_25z_group_11

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project supports data generation, training, and testing for 
the **Reber Grammar**, **Adding**, and **Multiplication** experiments using LRU, LSTM and GRU models.

## Requirements

- Python installed and available as `python`
- All required Python dependencies installed
- `make` available in your system

## Installing dependencies
```bash
make requirements
```

## Data Generators

In order to run generating raw data file for Reber experiment, run:

```bash
make reber_data
```

In order to run generating raw data file for adding and multiplication experiment, run:

```bash
make adding_data NUM=10000 RNG=42
make multiplication_data NUM=10000 RNG=42
```
#### Arguments:
`NUM`: Number of sequences to generate (integer)

`RNG`: Random RNG (integer)

## Experiments

### Reber experiment
#### Usage
```bash
make reber_exp MODEL=LSTM NUM_TESTS=1
```
#### Arguments:
`MODEL`: Model to use
* `LSTM` PyTorch implementation of long short-term memory (LSTM) RNN 
* `GRU`  PyTorch implementation of gated recurrent unit (GRU) RNN
* `LRU`  Gothos/LRU-pytorch implementation of Linear Recurrent Units (LRU)

`NUM_TESTS`: Number of tests to run (default value - 1)

### Adding experiment
#### Usage
```bash
make adding_exp MODEL=LSTM DATA=generate RNG=42
```
#### Arguments:
`MODEL`: Model to use</br>
* `LSTM` PyTorch implementation of long short-term memory (LSTM) RNN </br>
* `GRU`  PyTorch implementation of gated recurrent unit (GRU) RNN </br>
* `LRU`  Gothos/LRU-pytorch implementation of Linear Recurrent Units (LRU)

`DATA`: Data source</br>
* `generate` generate new sequences while learning</br>
* `file`     load sequences from file

`RNG`: Random RNG (integer)

### Multiplication experiment
#### Usage
```bash
make multiplication_exp MODEL=LSTM DATA=generate RNG=42
```
#### Arguments:
`MODEL`: Model to use</br>
* `LSTM` PyTorch implementation of long short-term memory (LSTM) RNN </br>
* `GRU`  PyTorch implementation of gated recurrent unit (GRU) RNN </br>
* `LRU`  Gothos/LRU-pytorch implementation of Linear Recurrent Units (LRU)

`DATA`: Data source</br>
* `generate` generate new sequences while learning</br>
* `file`     load sequences from file

`RNG`: Random RNG (integer)

## Tests

### Usage

Reber:

```bash
make test_reber
```

All tests:
```bash
make test_all
```

Adding:

```bash
make test_adding
````

Multiplication:

```bash
make test_multiplication
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
│   ├── reber_generator_test.py
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

