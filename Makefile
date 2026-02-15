# This Makefile requires the following commands to be available:
# * python3
# * virtualenv

ifeq ($(PYTHON_VERSION),)
	PYTHON_VERSION=python3
endif

REQUIREMENTS_TXT:=requirements.txt
OS=$(shell lsb_release -si 2>/dev/null || uname)
PIP:="venv/bin/pip3"
PYTHON="venv/bin/python3"

.DEFAULT_GOAL := venv

.PHONY: clean pyclean

pyclean:
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name __pycache__ -delete
	@find . -name .DS_Store -delete
	@rm -rf *.egg-info build
	@rm -rf coverage.xml .coverage

clean: pyclean
	@rm -rf venv

venv:
	@$(PYTHON_VERSION) -m venv venv
	@$(PIP) install --upgrade pip
	@$(PIP) install -r $(REQUIREMENTS_TXT)

run: venv
	@$(PYTHON) src/main.py
