#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = zprp_project_25z_group_11
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

#################################################################################
# DATA GENERATION                                                                #
#################################################################################

# Generate Reber raw data
reber_data:
	$(PYTHON_INTERPRETER) ./zprp_project_25z_group_11/generators/reber.py

# Generate Adding or Multiplication data
# Usage: make adding_data NUM=1000 RNG=42
adding_data:
	$(PYTHON_INTERPRETER) ./zprp_project_25z_group_11/generators/components.py --task adding --num_sequences $(NUM) --rng $(RNG)

multiplication_data:
	$(PYTHON_INTERPRETER) ./zprp_project_25z_group_11/generators/components.py --task multiplication --num_sequences $(NUM) --rng $(RNG)

#################################################################################
# EXPERIMENTS                                                                      #
#################################################################################

# Reber experiment
# Usage: make reber_exp MODEL=LSTM NUM_TESTS=1
reber_exp:
	$(PYTHON_INTERPRETER) ./zprp_project_25z_group_11/experiments/reber.py --model $(MODEL) --num-tests $(NUM_TESTS)

# Adding experiment
# Usage: make adding_exp MODEL=LSTM DATA=generate RNG=42
adding_exp:
	$(PYTHON_INTERPRETER) ./zprp_project_25z_group_11/experiments/adding.py --model $(MODEL) --data $(DATA) --rng $(RNG)

# Multiplication experiment
# Usage: make multiplication_exp MODEL=LSTM DATA=generate RNG=42
multiplication_exp:
	$(PYTHON_INTERPRETER) ./zprp_project_25z_group_11/experiments/multiplication.py --model $(MODEL) --data $(DATA) --rng $(RNG)

#################################################################################
# TESTS                                                                      #
#################################################################################

test_reber:
	pytest ./tests/reber_generator_test.py

test_adding:
	pytest ./tests/components_adding.py

test_multiplication:
	pytest ./tests/components_multiplication.py

test_all:
	pytest ./...

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) zprp_project_25z_group_11/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
