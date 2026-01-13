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
py .\zprp_project_25z_group_11\generators\components.py --task TASK_NAME --num_sequences N
```
#### Arguments:
`--task`: Task to generate data for</br>
* `adding`</br>
* `multiplication` 

`--num_sequences`: Number of sequences to generate (integer)</br>

`--rng`: Random RNG (integer)

## Experiments

### Adding experiment
#### Usage
```bash
python .\zprp_project_25z_group_11\experiments\adding.py --model MODEL_NAME --data DATA_MODE
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
python .\zprp_project_25z_group_11\experiments\multiplication.py --model MODEL_NAME --data DATA_MODE
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
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         zprp_project_25z_group_11 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── zprp_project_25z_group_11   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes zprp_project_25z_group_11 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

